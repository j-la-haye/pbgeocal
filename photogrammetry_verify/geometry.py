"""
#### geometry.py


The "Engine" of the package. It contains the mathematical logic for verification.
"""

import numpy as np

def world_to_image(world_pts, T_world_body, T_body_cam, camera):
    """Calculates 2D projections and validity masks."""
    # Chain: World -> Body -> Camera
    T_cam_world = T_body_cam.inverse().apply(T_world_body.inverse().apply(world_pts))

    depths = T_cam_world[:, 2]
    valid = depths > 0

    # Normalize and distort
    x_n = T_cam_world[:, 0] / np.where(valid, depths, 1.0)
    y_n = T_cam_world[:, 1] / np.where(valid, depths, 1.0)

    # Distortion (Radial k1, k2, k3 and Tangential p1, p2)
    k1, k2, p1, p2, k3 = camera.dist
    r2 = x_n**2 + y_n**2
    radial = 1 + k1*r2 + k2*(r2**2) + k3*(r2**3)

    x_d = x_n * radial + 2*p1*x_n*y_n + p2*(r2 + 2*x_n**2)
    y_d = y_n * radial + p1*(r2 + 2*y_n**2) + 2*p2*x_n*y_n

    # Apply K matrix
    u = camera.K[0,0] * x_d + camera.K[0,2]
    v = camera.K[1,1] * y_d + camera.K[1,2]

    return np.stack([u, v], axis=1), valid

def compute_ray_distance(img_pts, world_pts, T_wb, T_bc, camera):
    """Calculates L2 distance from 3D points to back-projected rays."""
    u, v = img_pts[:, 0], img_pts[:, 1]

    # Pixel to Camera Ray (Z=1)
    rays_c = np.stack([
        (u - camera.K[0,2]) / camera.K[0,0],
        (v - camera.K[1,2]) / camera.K[1,1],
        np.ones_like(u)
    ], axis=1)
    rays_c /= np.linalg.norm(rays_c, axis=1, keepdims=True)

    # Ray and Camera Origin in World Frame
    rays_w = T_wb.r.apply(T_bc.r.apply(rays_c))
    cam_w = T_wb.apply(T_bc.t)

    # Point-to-line distance
    vec_p = world_pts - cam_w
    return np.linalg.norm(np.cross(rays_w, vec_p), axis=1)
