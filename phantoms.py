import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from scipy.ndimage import rotate

def rotate_image(img, angle):
    return rotate(img, angle, reshape=False, order=1, mode="constant", cval=0.0)

def get_all_phantoms(size=32):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    rect = ((np.abs(x) < 0.7) & (np.abs(y) < 0.4)).astype(float)
    h_sq = np.zeros((size, size)); s = size // 4
    h_sq[s:3*s, s:3*s] = 1; h_sq[s+size//10:3*s-size//10, s+size//10:3*s-size//10] = 0
    tri = ((y > -0.5) & (y < -2*x + 0.5) & (y < 2*x + 0.5)).astype(float)
    cross = (((np.abs(x) < 0.1) & (np.abs(y) < 1)) | ((np.abs(y) < 0.1) & (np.abs(x) < 1))).astype(float)
    idx_x, idx_y = np.indices((size, size))
    check = (((idx_x // 2 + idx_y // 2) % 2) * (np.abs(x) < 0.5) * (np.abs(y) < 0.5)).astype(float)
    stripes = ((np.sin(x * 60) > 0) * (np.abs(x) < 0.5) * (np.abs(y) < 0.5)).astype(float)
    grad = (np.clip(x + 0.5, 0, 1) * (np.abs(x) < 0.5) * (np.abs(y) < 0.5)).astype(float)
    sl = resize(shepp_logan_phantom(), (size, size), mode="reflect", anti_aliasing=True)
    return [("Rectangle", rect), ("Hollow Square", h_sq), ("Triangle", tri), ("Cross", cross),
            ("Checkerboard", check), ("Stripes", stripes), ("Gradient", grad), ("Shepp-Logan", sl)]