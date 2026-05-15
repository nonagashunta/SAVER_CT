import numpy as np
from phantoms import get_all_phantoms, rotate_image
from saver_env import get_circle_mask
from ct_methods import run_ct_method
from const import * 

a_list = np.load("a_list_ratio0.25_mode_NULL_.npy")
square_mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)
eval_mask = get_circle_mask(square_mask)
phantoms = get_all_phantoms(IMG_SIZE)
methods = ["MAX-V", "MIN-V", "Random", "AIRS", "SAVER", "SAVER-O", "SAVER-A", "SAVER-AO"]

results = {}
for p_name, p_img in phantoms:
    print(f"Phantom: {p_name}")
    results[p_name] = {}
    for m_name in methods:
        print(f"  Method: {m_name}")
        trial_data = []
        for run in range(NUM_RUNS):
            seed = run + 100
            img_rot = rotate_image(p_img, np.random.RandomState(seed).randint(0, 46))
            ssim_hist = run_ct_method(m_name, a_list, img_rot, square_mask, eval_mask, 
                                      TOTAL_SHOTS, SIGMA_B, XI, ANGLE_NUM, 1, seed)
            trial_data.append(ssim_hist)
        results[p_name][m_name] = trial_data

np.save(f"final_results_{SIGMA_B}_{ETA}_{NUM_RUNS}.npy", results)
print("Saved to final_results.npy")