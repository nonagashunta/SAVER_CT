import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_roi_ssim(recon_img, gt_img, roi_mask):
    """論文 Eq.(5) 準拠：円内のみを平均するSSIM計算"""
    data_range = max(gt_img.max() - gt_img.min(), 1e-10)
    _, ssim_map = ssim(gt_img, recon_img, data_range=data_range, full=True)
    return np.mean(ssim_map[roi_mask.astype(bool)])

def get_circle_mask(square_mask):
    """評価領域 Omega_circle の生成"""
    ys, xs = np.where(square_mask)
    cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    r = (ys.max() - ys.min() + 1) / 2.0
    H, W = square_mask.shape
    Y, X = np.ogrid[:H, :W]
    return ((Y - cy)**2 + (X - cx)**2 <= r**2) & square_mask

class WoodburyEngine:
    def __init__(self, PIX_ROI, xi, sigma_B):
        self.x_est = np.zeros((PIX_ROI, 1))
        self.B_inv = (xi**2 / (sigma_B**2 + 1e-12)) * np.eye(PIX_ROI)
    def update(self, at, y):
        denom = float(1.0 + at.T @ self.B_inv @ at)
        K = (self.B_inv @ at) / denom
        self.x_est += K * (y - at.T @ self.x_est)
        self.B_inv -= K @ (at.T @ self.B_inv)