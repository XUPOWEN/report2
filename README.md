# 不平衡分類：信用卡詐欺偵測
南華大學跨領域-人工智慧期末報告
11124127王星圍 11124128蘇佑庭 11124130邱述陽
# 介紹
此範例查看 Kaggle 信用卡詐欺偵測 資料集，示範如何針對具有高度不平衡類別的資料訓練分類模型。連結:https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 首先，對 CSV 資料進行向量化
```
import csv
import numpy as np
     
fname = "/creditcard.csv"
     
all_features = []
all_targets = []
with open(fname) as f:
    for i, line in enumerate(f):
        if i == 0:
            print("HEADER:", line.strip())
            continue  # Skip header
        fields = line.strip().split(",")
        all_features.append([float(v.replace('"', "")) for v in fields[:-1]])
        all_targets.append([int(fields[-1].replace('"', ""))])
        if i == 1:
            print("EXAMPLE FEATURES:", all_features[-1])

features = np.array(all_features, dtype="float32")
targets = np.array(all_targets, dtype="uint8")
print("features.shape:", features.shape)
print("targets.shape:", targets.shape)
```
