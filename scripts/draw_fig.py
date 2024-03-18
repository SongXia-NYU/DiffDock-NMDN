import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

for pp_idx in [387, 388, 389, 390]:
    pred_df: pd.DataFrame = pd.read_csv(f"/scratch/sx801/shared/xuhang/pp_test_mdn_scores/pp_{pp_idx}.csv")
    dist = sns.histplot(pred_df, x="score")
    plt.savefig(f"./pp_{pp_idx}_dist.png")
    plt.close()
