import numpy as np
import random
from saver_env import WoodburyEngine, compute_roi_ssim

def _robust_norm(x):
    q75, q25 = np.percentile(x, [75, 25]); iqr = q75 - q25
    return (x - np.median(x)) / (iqr + 1e-12)

def _stable_softmax(logits, temp):
    shifted = (logits / temp) - np.max(logits / temp)
    exp_l = np.exp(shifted)
    return exp_l / np.sum(exp_l)

def run_ct_method(name, a_list, img_rot, square_mask, eval_roi_mask, 
                  total_shots, sigma_B, xi, angle_num, eval_interval, seed, anneal_rate=0.01):
    np.random.seed(seed); random.seed(seed)
    PRJ_NUM = len(a_list) // 180
    angle_vals = list(range(0, 180, angle_num))
    num_angles = len(angle_vals)
    img_vec_roi = img_rot[square_mask].reshape(-1, 1)
    engine = WoodburyEngine(np.sum(square_mask), xi, sigma_B)
    ssim_list, history_y, ray_ptr, shot_count = [], {i: [] for i in range(num_angles)}, np.zeros(num_angles, dtype=int), 0

    # 真の分散 (Oracle用データ)
    true_vars = None
    if name in ["MAX-V", "MIN-V", "SAVER-O", "SAVER-AO"]:
        true_vars = np.array([np.var([float(a_list[ang*PRJ_NUM+r].reshape(1,-1) @ img_vec_roi) for r in range(PRJ_NUM)]) for ang in angle_vals])

    # レイ順序生成 (Random)
    ray_order = []
    for i in range(num_angles):
        order = np.arange(PRJ_NUM); np.random.shuffle(order); ray_order.append(order)

    def execute_shot(idx):
        nonlocal shot_count
        if ray_ptr[idx] >= PRJ_NUM: return False
        r = ray_order[idx][ray_ptr[idx]]; at = a_list[angle_vals[idx]*PRJ_NUM+r].reshape(-1, 1)
        y = at.T @ img_vec_roi + np.random.normal(0, sigma_B)
        engine.update(at, y); history_y[idx].append(float(y)); ray_ptr[idx] += 1; shot_count += 1
        if shot_count % eval_interval == 0 or shot_count == total_shots:
            full = np.zeros_like(img_rot); full[square_mask] = engine.x_est.flatten()
            ssim_list.append(compute_roi_ssim(full, img_rot, eval_roi_mask))
        return True

    # --- Step 1: 0/90 優先初期化 (AIRS, SAVER-A, SAVER-AO のみ) ---
    if name in ["AIRS", "SAVER-A", "SAVER-AO"]:
        for idx in [i for i, v in enumerate(angle_vals) if v in [0, 90]]:
            while shot_count < total_shots and execute_shot(idx): pass

    # --- Step 2: 真の分散による順次完食 (MAX-V, MIN-V のみ) ---
    if name in ["MAX-V", "MIN-V"]:
        for idx in (np.argsort(true_vars)[::-1] if name=="MAX-V" else np.argsort(true_vars)):
            while shot_count < total_shots and execute_shot(idx): pass
        return ssim_list

    # --- Step 3: 各角度 2ショットの初期探索 (実用 SAVER / SAVER-A のみ) ---
    # ※ 論文通り SAVER-O / AO (Oracle) は既に真の分散を知っているため、このフェーズは不要
    if name in ["SAVER", "SAVER-A"]:
        for idx in range(num_angles):
            # SAVER-A の場合、0/90 は Step 1 で終わっているので飛ばす
            if name == "SAVER-A" and angle_vals[idx] in [0, 90]: continue
            for _ in range(2):
                if shot_count < total_shots: execute_shot(idx)

    # --- Step 4: Softmax 適応フェーズ (すべての SAVER系 & Random系) ---
    t_step = 0
    while shot_count < total_shots:
        avail = [i for i in range(num_angles) if ray_ptr[i] < PRJ_NUM]
        if not avail: break
        
        if name in ["Random", "AIRS"]: 
            idx = np.random.choice(avail)
        else: # SAVER系 (SAVER, SAVER-A, SAVER-O, SAVER-AO)
            # Oracle系は真の分散 true_vars を、実用系は推定分散(np.var)を使用
            v = true_vars[avail] if "-O" in name else np.array([np.var(history_y[i]) if len(history_y[i])>=2 else 0 for i in avail])
            probs = _stable_softmax(_robust_norm(v), max(0.1, 1.0/(1 + anneal_rate * t_step)))
            idx = np.random.choice(avail, p=probs)
        
        execute_shot(idx); t_step += 1
        
    return ssim_list