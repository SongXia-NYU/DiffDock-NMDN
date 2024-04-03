import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ic50_ds = pd.read_csv("/vast/sx801/geometries/Yaowen_nonbinders/ic50.csv")
sns.histplot(ic50_ds["Activity"])
plt.savefig("./scripts/ds_prepare/yaowen_nonbinders/ic50_ds.png")
plt.close()
print((ic50_ds["Unit"]!="neg. log").sum())
print(ic50_ds[ic50_ds["Unit"] != "neg. log"][["Unit"]].isna().sum())
print("Total number of entries in IC50: ", ic50_ds.shape[0])
print("Total number of entries <= 4.0: ", (ic50_ds["Activity"] <= 4.0).sum())
print("-"*40)

ki_ds = pd.read_csv("/vast/sx801/geometries/Yaowen_nonbinders/ki.csv")
sns.histplot(ki_ds["Activity"])
plt.savefig("./scripts/ds_prepare/yaowen_nonbinders/ki_ds.png")
plt.close()
print((ki_ds["Unit"]!="neg. log").sum())
print(ki_ds[ki_ds["Unit"] != "neg. log"][["Unit"]].isna().sum())
print("Total number of entries in Ki: ", ki_ds.shape[0])
print("Total number of entries <= 4.0: ", (ki_ds["Activity"] <= 4.0).sum())
print("-"*40)
