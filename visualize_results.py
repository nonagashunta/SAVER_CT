import numpy as np
import matplotlib.pyplot as plt
import os
from const import *

# ==========================================
# 0. データの読み込みと準備
# ==========================================
# 実験結果ファイルをロード
# FILE_PATH = f"final_results_{SIGMA_B}_{ETA}_{NUM_RUNS}.npy"
# FILE_PATH = f"results_sigma1e-02_eta1.0.npy"
# FILE_PATH = f"results_sigma1e-02_eta0.1.npy"
FILE_PATH = f"results_sigma1e-02_eta0.01.npy"
# FILE_PATH = f"results_sigma1e-06_eta1.0.npy"
# FILE_PATH = f"results_sigma1e-06_eta0.1.npy"
# FILE_PATH = f"results_sigma1e-06_eta0.01.npy"
if not os.path.exists(FILE_PATH):
    print(f"Error: {FILE_PATH} が見つかりません。先に実験を実行してください。")
    exit()

results_storage = np.load(FILE_PATH, allow_pickle=True).item()

# 基本設定
total_shots = 500
eval_steps = np.arange(1, total_shots + 1)

# 箱ひげ図用のデータを SSIM履歴の合計から計算
auc_data_all = {}
for p_name, methods in results_storage.items():
    auc_data_all[p_name] = {}
    for m_name, trial_data in methods.items():
        # 各試行ごとのSSIMの合計値を計算 (AUC算出用)
        auc_data_all[p_name][m_name] = [np.sum(run) for run in trial_data]

# ==========================================
# 1. 可視化：SSIM推移グラフ (2行5列)
# ==========================================
plt.rcParams["font.family"] = "DejaVu Serif"   # 使用するフォント
plt.rcParams["font.size"] = 15     

# 2行5列の設定
fig1, axes1 = plt.subplots(2, 5, figsize=(17.5, 7), dpi=600)
axes_flat = axes1.flatten()

# 1. 設定リストを (データのキー, 表示ラベル, 色, スタイル, 線幅) の形式
methods_plot_cfg = [
    ("MAX-V", "MAX-V", "r", "--", 0.8),
    ("MIN-V", "MIN-V", "b", "--", 0.8),
    ("Random", "Random", "k", "-", 0.8),
    ("AIRS", "AIRS", "orange", "-", 1),
    ("SAVER", r"SAVER ($\eta$=0.01)", "m", "-", 1.2),
    ("SAVER-O", r"SAVER-O ($\eta$=0.01)", "m", ":", 1.2),
    ("SAVER-A", r"SAVER-A ($\eta$=0.01)", "g", "-", 1.2),
    ("SAVER-AO", r"SAVER-AO ($\eta$=0.01)", "g", ":", 1.2),
]

for i, (p_name, methods) in enumerate(results_storage.items()):
    if i >= 8: break
    ax = axes_flat[i]

    # 2. ループ内で 5つの変数を受け取る
    for m_key, m_label, color, style, lw in methods_plot_cfg:
        if m_key in methods:
            data = np.array(methods[m_key])
            if data.ndim > 2:
                data = data[:, :, 0]

            mean_vals = np.mean(data, axis=0)
            std_vals = np.std(data, axis=0)

            ax.plot(
                eval_steps,
                mean_vals,
                label=m_label,
                color=color,
                linestyle=style,
                lw=lw
            )
            ax.fill_between(eval_steps, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.07)

    # タイトルや軸の設定
    ax.set_title(f"{i+1}. {p_name}", fontsize=20)
    ax.set_xlabel("Round")
    ax.set_ylabel("SSIM")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()

# --- 凡例表示エリアの処理 ---
mid = len(handles) // 2
handles1, labels1 = handles[:mid], labels[:mid]
handles2, labels2 = handles[mid:], labels[mid:]

# 9番目の枠 (index 8)
axes_flat[8].axis('off')
if handles1:
    axes_flat[8].legend(handles1, labels1, loc='center', fontsize=20, frameon=False, title="Methods (1/2)")

# 10番目の枠 (index 9)
axes_flat[9].axis('off')
if handles2:
    axes_flat[9].legend(handles2, labels2, loc='center', fontsize=20, frameon=False, title="Methods (2/2)")

# 未使用の枠を非表示
for j in range(len(results_storage), 8):
    axes_flat[j].axis('off')

plt.tight_layout()
plt.savefig(f"ssim_curves_{SIGMA_B}_{ETA}_{NUM_RUNS}.png", bbox_inches='tight') # ファイル保存
print("Saved: ssim_curves.png")
plt.close(fig1)

# ==========================================
# 2. 可視化：積分値(AUC)の箱ひげ図 (2行4列)
# ==========================================
fig2, axes2 = plt.subplots(2, 4, figsize=(14, 4.8), dpi=600)
unified_color = '#e3f2fd' # 淡い青色

for i, (p_name, m_aucs) in enumerate(auc_data_all.items()):
    if i >= 8: break
    ax = axes2.flatten()[i]
    labels = list(m_aucs.keys())
    
    # データ正規化 (合計 / 500)
    data = [np.array(m_aucs[l]) / 500 for l in labels]
    
    # 箱ひげ図の描画
    bp = ax.boxplot(data, patch_artist=True, 
                    medianprops=dict(color="orange", linewidth=1.5),
                    boxprops=dict(color="#333333", linewidth=1))
    
    for patch in bp['boxes']:
        patch.set_facecolor(unified_color)

    ax.set_title(f"{i+1}. {p_name}", fontsize=15)
    
    # X軸ラベル調整
    if i >= 4: 
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=11)
    else: 
        ax.set_xticklabels([]) 

    # Y軸ラベル調整 (左端のみ)
    if i % 4 == 0:
        ax.set_ylabel("AUC / 500", fontsize=12)
    else: 
        ax.set_ylabel("")

    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.15, 0.90)

plt.tight_layout()
plt.savefig(f"auc_boxplots_{SIGMA_B}_{ETA}_{NUM_RUNS}.png", bbox_inches='tight') # ファイル保存
print("Saved: auc_boxplots.png")
plt.close(fig2)

# 2行5列の設定
fig3, axes3 = plt.subplots(2, 5, figsize=(17.5, 7), dpi=600)
axes_flat = axes3.flatten()

# 1. 設定リストを (データのキー, 表示ラベル, 色, スタイル, 線幅) の形式
methods_plot_cfg = [
    # ("MAX-V", "MAX-V", "r", "--", 0.8),
    # ("MIN-V", "MIN-V", "b", "--", 0.8),
    ("Random", "Random", "k", "-", 0.8),
    ("AIRS", "AIRS", "orange", "-", 1),
    # ("SAVER", r"SAVER ($\eta$=0.01)", "m", "-", 1.2),
    # ("SAVER-O", r"SAVER-O ($\eta$=0.01)", "m", ":", 1.2),
    ("SAVER-A", r"SAVER-A ($\eta$=0.01)", "g", "-", 1.2),
    ("SAVER-AO", r"SAVER-AO ($\eta$=0.01)", "g", ":", 1.2),
]

for i, (p_name, methods) in enumerate(results_storage.items()):
    if i >= 8: break
    ax = axes_flat[i]

    # 2. ループ内で 5つの変数を受け取る
    for m_key, m_label, color, style, lw in methods_plot_cfg:
        if m_key in methods:
            data = np.array(methods[m_key])
            if data.ndim > 2:
                data = data[:, :, 0]

            mean_vals = np.mean(data, axis=0)
            std_vals = np.std(data, axis=0)

            ax.plot(
                eval_steps,
                mean_vals,
                label=m_label,
                color=color,
                linestyle=style,
                lw=lw
            )
            ax.fill_between(eval_steps, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.07)

    # タイトルや軸の設定
    ax.set_title(f"{i+1}. {p_name}", fontsize=20)
    ax.set_xlabel("Round")
    ax.set_ylabel("SSIM")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()

# --- 凡例表示エリアの処理 ---
mid = len(handles) // 2
handles1, labels1 = handles[:mid], labels[:mid]
handles2, labels2 = handles[mid:], labels[mid:]

# 9番目の枠 (index 8)
axes_flat[8].axis('off')
if handles1:
    axes_flat[8].legend(handles, labels, loc='center', fontsize=20, frameon=False, title="Methods")

# 10番目の枠 (index 9)
axes_flat[9].axis('off')
# if handles2:
#     axes_flat[9].legend(handles2, labels2, loc='center', fontsize=20, frameon=False, title="Methods (2/2)")

# 未使用の枠を非表示
for j in range(len(results_storage), 8):
    axes_flat[j].axis('off')

plt.tight_layout()
plt.savefig(f"ssim_curves_axis_{SIGMA_B}_{ETA}_{NUM_RUNS}.png", bbox_inches='tight') # ファイル保存
print("Saved: ssim_curves_axis.png")
plt.close(fig3)