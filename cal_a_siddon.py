import numpy as np
from const import *
from structure import VEC2D
from tqdm import tqdm
from siddon import get_siddon_weights
import math

def rotate_point(pt, theta):
    # const.py の CLOCKWISE (-1) を考慮
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # 原点周りの回転
    new_x = cos_t * pt.x - CLOCKWISE * sin_t * pt.y
    new_y = CLOCKWISE * sin_t * pt.x + cos_t * pt.y
    return VEC2D(new_x, new_y)

def calculate_system_matrix_siddon():
    radius = IMG_SIZE / 2.0
    center = IMG_SIZE / 2.0
    
    # 最終的に (ANG_NUM * PRJ_NUM, PIX_NUM) の行列を作りたい
    # メモリ節約のためリストで保持
    a_list = []

    for j in tqdm(range(ANG_NUM), desc="Angles"):
        theta = j * math.pi / ANG_NUM
        
        for t in range(PRJ_NUM):
            # レイの検出器上の位置 (中心を0とする)
            # joseph.py の calcCoordXrayAndDtc のロジックに合わせる
            # ratio=1.0 と仮定 (または 0.25 の場合はその分サンプリングが必要)
            # ここでは単純化のため1つの検出器スロットにつき1本のレイで計算
            det_pos = t - radius + 0.5
            
            # 回転前の始点と終点
            p1_unrot = VEC2D(det_pos, -radius - 1.0)
            p2_unrot = VEC2D(det_pos, radius + 1.0)
            
            # 回転
            p1_rot = rotate_point(p1_unrot, theta)
            p2_rot = rotate_point(p2_unrot, theta)
            
            # 画像座標系 (0 ~ IMG_SIZE) へ変換
            p1 = VEC2D(p1_rot.x + center, p1_rot.y + center)
            p2 = VEC2D(p2_rot.x + center, p2_rot.y + center)
            
            # Siddon法でこのレイが通るピクセルと長さを取得
            weights = get_siddon_weights(p1, p2)
            
            # システム行列の1行分 (ベクトル a)
            row_a = np.zeros(PIX_NUM)
            for idx, length in weights:
                row_a[idx] = length
            
            a_list.append(row_a.reshape(-1, 1))

    # 保存
    a_list_np = np.array(a_list)
    print(f"Matrix shape: {a_list_np.shape}")
    np.save(f"a_list_siddon_{IMG_SIZE}.npy", a_list_np)

if __name__ == "__main__":
    calculate_system_matrix_siddon()