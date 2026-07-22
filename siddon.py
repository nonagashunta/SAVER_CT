import numpy as np
import math
from const import IMG_SIZE

def get_siddon_weights(p1, p2):
    """
    Siddon's algorithm to find intersection lengths with pixels.
    p1, p2: VEC2D (start and end points in pixel coordinates)
    Returns: list of (pixel_index, length)
    """
    nx, ny = IMG_SIZE, IMG_SIZE
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    # パラメータ alpha の最小値と最大値を計算 (グリッド内に入る範囲)
    # グリッドの範囲は 0.0 ~ nx (ピクセル境界)
    alphas_x = []
    if abs(dx) > 1e-9:
        vx = [float(i) for i in range(nx + 1)]
        alphas_x = [(v - p1.x) / dx for v in vx]
    
    alphas_y = []
    if abs(dy) > 1e-9:
        vy = [float(j) for j in range(ny + 1)]
        alphas_y = [(v - p1.y) / dy for v in vy]

    # 有効な alpha (0.0 から 1.0 の間) を抽出してソート
    alpha_list = [0.0, 1.0]
    for a in alphas_x:
        if 0.0 < a < 1.0: alpha_list.append(a)
    for a in alphas_y:
        if 0.0 < a < 1.0: alpha_list.append(a)
    
    alpha_list = sorted(list(set(alpha_list)))

    results = []
    # 全体の長さ
    total_len = math.sqrt(dx**2 + dy**2)

    for i in range(len(alpha_list) - 1):
        a_mid = (alpha_list[i] + alpha_list[i+1]) / 2.0
        # 中間点の座標からピクセルインデックスを特定
        px = int(math.floor(p1.x + a_mid * dx))
        py = int(math.floor(p1.y + a_mid * dy))

        if 0 <= px < nx and 0 <= py < ny:
            length = (alpha_list[i+1] - alpha_list[i]) * total_len
            pixel_idx = py * nx + px
            results.append((pixel_idx, length))
            
    return results