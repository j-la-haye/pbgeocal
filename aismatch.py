import os
import yaml
import cv2
import torch
import numpy as np
from spectral import open_image
from tqdm import tqdm
from PIL import Image

# LoFTR (original) – requires the LoFTR repo in PYTHONPATH
import sys
sys.path.append("/media/lasigadmin/BCFE4CF2FE4CA68C/tools/LoFTR/src")
from loftr import LoFTR
from loftr.utils.cvpr_ds_config import default_cfg

# EfficientLoFTR from Hugging Face Transformers
from transformers import AutoImageProcessor, AutoModelForKeypointMatching


# =====================================================
#                 UTILITIES
# =====================================================

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_envi_bil_as_gray(path, band_indices=None):
    """
    Load ENVI BIL/BIP/BSQ via spectral and collapse to a single uint8 grayscale image.
    """
    img = open_image(path)
    cube = np.array(img.load())  # (H, W, C)

    if band_indices:
        band_indices = list(band_indices)
        sel = cube[:, :, band_indices]
        gray = sel.mean(axis=2)
    else:
        gray = cube.mean(axis=2)

    gray = gray.astype(np.float32)
    mn, mx = gray.min(), gray.max()
    if mx > mn:
        gray = (gray - mn) / (mx - mn)
    else:
        gray = np.zeros_like(gray)
    gray = (gray * 255.0).clip(0, 255).astype(np.uint8)
    return gray


# =====================================================
#                 TILING
# =====================================================

def generate_tiles(img, tile_size, overlap):
    """
    Yield (x, y, tile) for a grid of overlapping tiles.
    """
    H, W = img.shape
    step = max(1, tile_size - overlap)

    tiles = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            tile = img[y:y_end, x:x_end]
            tiles.append((x, y, tile))
    return tiles


def run_tiled_matching(img1, img2, matcher_fn, tile_size, overlap, **kwargs):
    """
    Run a matcher function on corresponding tiles of img1 and img2.
    matcher_fn: callable(img1_tile, img2_tile, **kwargs) -> (pts1, pts2, conf)
    Returns concatenated (pts1, pts2, conf) in global coordinates.
    """
    tiles1 = generate_tiles(img1, tile_size, overlap)
    tiles2 = generate_tiles(img2, tile_size, overlap)
    assert len(tiles1) == len(tiles2), "Tiling mismatch between images."

    all_pts1, all_pts2, all_conf = [], [], []

    for (x1, y1, t1), (x2, y2, t2) in tqdm(
        list(zip(tiles1, tiles2)), desc="Tiled matching"
    ):
        # assuming same tiling for both images
        try:
            pts1, pts2, conf = matcher_fn(t1, t2, **kwargs)
            if pts1.shape[0] == 0:
                continue

            # offset back to global coordinates
            pts1 = pts1.copy()
            pts2 = pts2.copy()
            pts1[:, 0] += x1
            pts1[:, 1] += y1
            pts2[:, 0] += x1
            pts2[:, 1] += y1

            all_pts1.append(pts1)
            all_pts2.append(pts2)
            all_conf.append(conf)
        except Exception as e:
            # skip tiles that fail
            continue

    if not all_pts1:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.concatenate(all_pts1, axis=0),
        np.concatenate(all_pts2, axis=0),
        np.concatenate(all_conf, axis=0),
    )


# =====================================================
#                 LOFTR (ORIGINAL)
# =====================================================

def preprocess_for_loftr(gray_img):
    t = torch.from_numpy(gray_img).float() / 255.0
    return t.unsqueeze(0).unsqueeze(0)


def run_loftr(img1_gray, img2_gray, weights_path, device):
    """
    Original LoFTR from the ZJU repo (src.loftr).
    """
    t1 = preprocess_for_loftr(img1_gray).to(device)
    t2 = preprocess_for_loftr(img2_gray).to(device)
    data = {"image0": t1, "image1": t2}

    matcher = LoFTR(config=default_cfg)
    ckpt = torch.load(weights_path, map_location=device)
    matcher.load_state_dict(ckpt["state_dict"])
    matcher = matcher.to(device).eval()

    with torch.no_grad():
        matcher(data)

    pts1 = data["mkpts0_f"].cpu().numpy()
    pts2 = data["mkpts1_f"].cpu().numpy()
    conf = data["mconf"].cpu().numpy()
    return pts1, pts2, conf


# =====================================================
#             EFFICIENT LOFTR (TRANSFORMERS)
# =====================================================

def run_efficientloftr(
    img1_gray,
    img2_gray,
    model_name="zju-community/efficientloftr",
    threshold=0.2,
    device="cuda",
):
    """
    EfficientLoFTR via Hugging Face Transformers AutoModelForKeypointMatching.
    Runs on the full images (no tiling, per your Option C).
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForKeypointMatching.from_pretrained(model_name).to(device).eval()

    img1_pil = Image.fromarray(img1_gray)
    img2_pil = Image.fromarray(img2_gray)
    images = [img1_pil, img2_pil]

    inputs = processor(images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Use post_process_keypoint_matching to map to original image coords
    image_sizes = [[(img.height, img.width) for img in images]]
    processed = processor.post_process_keypoint_matching(
        outputs, image_sizes, threshold=threshold
    )

    d = processed[0]
    pts1 = d["keypoints0"].cpu().numpy().astype(np.float32)
    pts2 = d["keypoints1"].cpu().numpy().astype(np.float32)
    conf = d["matching_scores"].cpu().numpy().astype(np.float32)
    return pts1, pts2, conf


# =====================================================
#               ORB & SURF (OpenCV)
# =====================================================

def run_orb(img1_gray, img2_gray, n_features=5000, **kwargs):
    orb = cv2.ORB_create(nfeatures=n_features)
    k1, d1 = orb.detectAndCompute(img1_gray, None)
    k2, d2 = orb.detectAndCompute(img2_gray, None)

    if d1 is None or d2 is None:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if not matches:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    pts1 = np.array([k1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([k2[m.trainIdx].pt for m in matches], dtype=np.float32)
    dist = np.array([m.distance for m in matches], dtype=np.float32)
    conf = 1.0 / (dist + 1e-8)
    return pts1, pts2, conf


def run_surf(img1_gray, img2_gray, hessian_threshold=400, **kwargs):
    # Requires opencv-contrib-python
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)

    k1, d1 = surf.detectAndCompute(img1_gray, None)
    k2, d2 = surf.detectAndCompute(img2_gray, None)

    if d1 is None or d2 is None:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(d1, d2)
    if not matches:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    pts1 = np.array([k1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([k2[m.trainIdx].pt for m in matches], dtype=np.float32)
    dist = np.array([m.distance for m in matches], dtype=np.float32)
    conf = 1.0 / (dist + 1e-8)
    return pts1, pts2, conf


# =====================================================
#               METRICS & VISUALIZATION
# =====================================================

def compute_metrics(pts1, pts2, conf):
    metrics = {}
    N = pts1.shape[0]
    metrics["num_matches"] = int(N)
    metrics["mean_confidence"] = float(conf.mean()) if N > 0 else 0.0

    if N < 4:
        metrics["inlier_ratio"] = 0.0
        metrics["reprojection_error"] = -1.0
        return metrics

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        metrics["inlier_ratio"] = 0.0
        metrics["reprojection_error"] = -1.0
        return metrics

    inliers = mask.ravel().astype(bool)
    if N > 0:
        metrics["inlier_ratio"] = float(inliers.mean())
    else:
        metrics["inlier_ratio"] = 0.0

    pts1_proj = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H).reshape(-1, 2)
    reproj = np.linalg.norm(pts2 - pts1_proj, axis=1)
    metrics["reprojection_error"] = float(reproj.mean())
    return metrics


def draw_matches(img1, img2, pts1, pts2, conf, name, out_dir, top_k=300):
    if pts1.shape[0] == 0:
        print(f"No matches for {name}, skipping visualization.")
        return

    idx = np.argsort(-conf)
    idx = idx[: min(top_k, idx.shape[0])]
    pts1 = pts1[idx]
    pts2 = pts2[idx]

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1 + w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for p1, p2 in zip(pts1, pts2):
        x0, y0 = int(round(p1[0])), int(round(p1[1]))
        x1, y1 = int(round(p2[0] + w1)), int(round(p2[1]))
        cv2.line(canvas, (x0, y0), (x1, y1), (0, 255, 0), 1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, name)
    cv2.imwrite(out_path, canvas)
    print(f"Saved {name} -> {out_path}")


# =====================================================
#                     MAIN
# =====================================================

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    img1_path = cfg["images"]["img1"]
    img2_path = cfg["images"]["img2"]
    bands1 = cfg["bands"]["img1"]
    bands2 = cfg["bands"]["img2"]

    img1 = load_envi_bil_as_gray(img1_path, bands1)
    img2 = load_envi_bil_as_gray(img2_path, bands2)

    out_dir = cfg["output"]["directory"]
    top_k_vis = cfg["output"].get("top_k_vis", 300)

    tiling_cfg = cfg.get("tiling", {})
    do_tiling = tiling_cfg.get("enabled", False)
    tile_size = tiling_cfg.get("tile_size", 512)
    overlap = tiling_cfg.get("overlap", 64)

    metrics_enabled = cfg.get("metrics", {}).get("enabled", True)
    algos = cfg["algorithms"]

    device = get_device()
    print(f"Using device: {device}")

    # Define runners per algorithm
    def loftr_runner(a, b, **kwargs):
        return run_loftr(a, b, weights_path=cfg["loftr"]["weights"], device=device)

    def effloftr_runner(a, b, **kwargs):
        # ignores tiling, always full image
        return run_efficientloftr(
            a,
            b,
            model_name=cfg["efficientloftr"]["model_name"],
            threshold=cfg["efficientloftr"]["threshold"],
            device=device,
        )

    def orb_runner(a, b, **kwargs):
        return run_orb(a, b, n_features=cfg["orb"]["n_features"])

    def surf_runner(a, b, **kwargs):
        return run_surf(a, b, hessian_threshold=cfg["surf"]["hessian_threshold"])

    runners = {
        "loftr": loftr_runner,
        "efficientloftr": effloftr_runner,
        "orb": orb_runner,
        "surf": surf_runner,
    }

    for name, enabled in algos.items():
        if not enabled:
            continue
        print(f"\n=== Running {name.upper()} ===")
        runner = runners[name]

        # EfficientLoFTR always full image (Option C)
        if name == "efficientloftr":
            pts1, pts2, conf = runner(img1, img2)
        else:
            if do_tiling:
                pts1, pts2, conf = run_tiled_matching(
                    img1, img2, runner, tile_size, overlap
                )
            else:
                pts1, pts2, conf = runner(img1, img2)

        if metrics_enabled:
            m = compute_metrics(pts1, pts2, conf)
            print(f"Metrics [{name}]: {m}")

        draw_matches(
            img1,
            img2,
            pts1,
            pts2,
            conf,
            f"matches_{name}.png",
            out_dir,
            top_k=top_k_vis,
        )


if __name__ == "__main__":
    main()
