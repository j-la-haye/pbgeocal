import torch
import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
from kornia_moons.viz import draw_LAF_matches

def load_and_preprocess(path, device, resize_to=(768, 768)):
    """Loads image, converts to grayscale tensor, and resizes."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to tensor [1, 1, H, W] and scale to [0, 1]
    timg = K.image_to_tensor(img, keepdim=False).float() / 255.0
    timg = K.color.rgb_to_grayscale(timg).to(device)
    
    # Pushbroom images are often large; resizing is usually necessary for GPU memory
    timg_resized = K.geometry.transform.resize(timg, resize_to, antialias=True)
    return timg_resized, timg, img

def run_matching(img_path1, img_path2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Images
    image1, orig_t1, raw1 = load_and_preprocess(img_path1, device)
    image2, orig_t2, raw2 = load_and_preprocess(img_path2, device)

    # --- METHOD 1: SuperPoint + LightGlue (Feature-Based) ---
    # Great for distinct objects and handling horizontal jitter
    extractor = KF.SuperPointDetector(pretrained=True).to(device).eval()
    matcher_lg = KF.LightGlue('superpoint').to(device).eval()

    with torch.inference_mode():
        # Extract features
        lafs1, res1 = extractor(image1)
        lafs2, res2 = extractor(image2)
        
        # Match features
        # Note: LightGlue expects a dict of features
        feats1 = {'keypoints': lafs1, 'descriptors': res1}
        feats2 = {'keypoints': lafs2, 'descriptors': res2}
        dists, idxs = matcher_lg(feats1, feats2)

    # --- METHOD 2: LoFTR (Detector-Free) ---
    # Better for low-texture areas common in 1m satellite imagery
    matcher_loftr = KF.LoFTR(pretrained='outdoor').to(device).eval()

    with torch.inference_mode():
        input_dict = {"image0": image1, "image1": image2}
        correspondences = matcher_loftr(input_dict)
        
    # Extract coordinates
    mkpts1 = correspondences['keypoints0'].cpu().numpy()
    mkpts2 = correspondences['keypoints1'].cpu().numpy()

    # --- Visualization ---
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot LightGlue (Showing Top 50 for clarity)
    ax[0].set_title("LightGlue Matches (Tie Points)")
    # (Visualization logic for LightGlue using kornia_moons or manual plotting)
    # Simple scatter for demonstration:
    ax[0].imshow(raw1)
    ax[0].scatter(mkpts1[:, 0], mkpts1[:, 1], s=1, c='r')
    
    # Plot LoFTR
    ax[1].set_title("LoFTR Dense Matches")
    ax[1].imshow(raw2)
    ax[1].scatter(mkpts2[:, 0], mkpts2[:, 1], s=1, c='g')
    
    plt.show()

if __name__ == "__main__":
    run_matching("pushbroom_left.tif", "pushbroom_right.tif")

