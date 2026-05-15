import numpy as np
import os
import time
from phantoms import get_all_phantoms, rotate_image
from saver_env import get_circle_mask
from ct_methods import run_ct_method

# ==========================================
# 1. 実験設定
# ==========================================
IMG_SIZE = 32
NUM_RUNS = 10
TOTAL_SHOTS = 500
ANGLE_NUM = 3
XI = 0.1
EVAL_INTERVAL = 1

# 実験対象のパラメータリスト
SIGMA_LIST = [1e-6, 1e-2]
ETA_LIST = [0.01, 0.1, 1.0]

# データのロード
A_LIST_PATH = "a_list_ratio0.25_mode_NULL_.npy"
if not os.path.exists(A_LIST_PATH):
    raise FileNotFoundError(f"{A_LIST_PATH} が見つかりません。")

a_list = np.load(A_LIST_PATH)
square_mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)
eval_mask = get_circle_mask(square_mask)
phantoms = get_all_phantoms(IMG_SIZE)
methods = ["MAX-V", "MIN-V", "Random", "AIRS", "SAVER", "SAVER-O", "SAVER-A", "SAVER-AO"]

# 出力ディレクトリの作成
OUTPUT_DIR = "experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. メイン実験ループ
# ==========================================
start_time = time.time()

for sigma_B in SIGMA_LIST:
    for eta in ETA_LIST:
        print(f"\n" + "="*50)
        print(f"STARTING: sigma_B={sigma_B}, eta={eta}")
        print("="*50)
        
        # この組み合わせの結果を格納する辞書
        current_results = {}
        
        for p_name, p_img in phantoms:
            print(f"\nPhantom: {p_name}")
            current_results[p_name] = {}
            
            for m_name in methods:
                print(f"  Method: {m_name} (sigma={sigma_B}, eta={eta})")
                trial_data = []
                
                for run in range(NUM_RUNS):
                    seed = run + 100
                    img_rot = rotate_image(p_img, np.random.RandomState(seed).randint(0, 46))
                    
                    # コアロジックの実行
                    ssim_hist = run_ct_method(
                        name=m_name,
                        a_list=a_list,
                        img_rot=img_rot,
                        square_mask=square_mask,
                        eval_roi_mask=eval_mask,
                        total_shots=TOTAL_SHOTS,
                        sigma_B=sigma_B,
                        xi=XI,
                        angle_num=ANGLE_NUM,
                        eval_interval=EVAL_INTERVAL,
                        seed=seed,
                        anneal_rate=eta  # ここで eta を渡す
                    )
                    trial_data.append(ssim_hist)
                
                current_results[p_name][m_name] = trial_data
        
        # パラメータごとにファイルを保存 (途中で止まってもデータが残るように)
        file_name = f"results_sigma{sigma_B:.0e}_eta{eta}.npy"
        save_path = os.path.join(OUTPUT_DIR, file_name)
        np.save(save_path, current_results)
        print(f"\nSuccessfully saved results to: {save_path}")

end_time = time.time()
elapsed = (end_time - start_time) / 60
print(f"\n" + "="*50)
print(f"ALL EXPERIMENTS COMPLETED in {elapsed:.2f} minutes.")
print("="*50)