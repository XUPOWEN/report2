# 不平衡分類：信用卡詐欺偵測
南華大學跨領域-人工智慧期末報告
11124127王星圍 11124128蘇佑庭 11124130邱述陽
# 介紹
此範例查看 Kaggle 信用卡詐欺偵測 資料集，示範如何針對具有高度不平衡類別的資料訓練分類模型。
## Kaggle連結:https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
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
```
輸出結果
HEADER: "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"
EXAMPLE FEATURES: [0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]
features.shape: (284807, 30)
targets.shape: (284807, 1)
```
# 準備驗證集
```
num_val_samples = int(len(features) * 0.2)
train_features = features[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_features = features[-num_val_samples:]
val_targets = targets[-num_val_samples:]

print("Number of training samples:", len(train_features))
print("Number of validation samples:", len(val_features))
```
```
輸出結果
Number of training samples: 227846
Number of validation samples: 56961
```
# 分析目標中的類別不平衡
```
counts = np.bincount(train_targets[:, 0])
print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(train_targets)
    )
)

weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]
```
```
輸出結果
Number of positive samples in training data: 417 (0.18% of total)
```
# 使用訓練集統計數據標準化數據
```
mean = np.mean(train_features, axis=0)
train_features -= mean
val_features -= mean
std = np.std(train_features, axis=0)
train_features /= std
val_features /= std
```
# 建構二元分類模型
```
import keras

model = keras.Sequential(
    [
        keras.Input(shape=train_features.shape[1:]),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()
```
```
輸出結果
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_20 (Dense)            (None, 256)               7936      
                                                                 
 dense_21 (Dense)            (None, 256)               65792     
                                                                 
 dropout_10 (Dropout)        (None, 256)               0         
                                                                 
 dense_22 (Dense)            (None, 256)               65792     
                                                                 
 dropout_11 (Dropout)        (None, 256)               0         
                                                                 
 dense_23 (Dense)            (None, 1)                 257       
                                                                 
=================================================================
Total params: 139777 (546.00 KB)
Trainable params: 139777 (546.00 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
# class_weight用參數訓練模型
```
metrics = [
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=metrics
)

callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.keras")]
class_weight = {0: weight_for_0, 1: weight_for_1}

model.fit(
    train_features,
    train_targets,
    batch_size=2048,
    epochs=30,
    verbose=2,
    callbacks=callbacks,
    validation_data=(val_features, val_targets),
    class_weight=class_weight,
)
```
```
輸出結果
Epoch 1/30
112/112 - 3s - loss: 2.3843e-06 - fn: 48.0000 - fp: 30665.0000 - tn: 196764.0000 - tp: 369.0000 - precision: 0.0119 - recall: 0.8849 - val_loss: 0.1229 - val_fn: 5.0000 - val_fp: 1849.0000 - val_tn: 55037.0000 - val_tp: 70.0000 - val_precision: 0.0365 - val_recall: 0.9333 - 3s/epoch - 28ms/step
Epoch 2/30
112/112 - 1s - loss: 1.5423e-06 - fn: 33.0000 - fp: 9176.0000 - tn: 218253.0000 - tp: 384.0000 - precision: 0.0402 - recall: 0.9209 - val_loss: 0.0647 - val_fn: 10.0000 - val_fp: 568.0000 - val_tn: 56318.0000 - val_tp: 65.0000 - val_precision: 0.1027 - val_recall: 0.8667 - 788ms/epoch - 7ms/step
Epoch 3/30
112/112 - 1s - loss: 1.1238e-06 - fn: 27.0000 - fp: 7390.0000 - tn: 220039.0000 - tp: 390.0000 - precision: 0.0501 - recall: 0.9353 - val_loss: 0.0762 - val_fn: 10.0000 - val_fp: 591.0000 - val_tn: 56295.0000 - val_tp: 65.0000 - val_precision: 0.0991 - val_recall: 0.8667 - 637ms/epoch - 6ms/step
Epoch 4/30
112/112 - 1s - loss: 1.2840e-06 - fn: 32.0000 - fp: 8359.0000 - tn: 219070.0000 - tp: 385.0000 - precision: 0.0440 - recall: 0.9233 - val_loss: 0.1417 - val_fn: 5.0000 - val_fp: 2768.0000 - val_tn: 54118.0000 - val_tp: 70.0000 - val_precision: 0.0247 - val_recall: 0.9333 - 647ms/epoch - 6ms/step
Epoch 5/30
112/112 - 1s - loss: 9.5530e-07 - fn: 26.0000 - fp: 7347.0000 - tn: 220082.0000 - tp: 391.0000 - precision: 0.0505 - recall: 0.9376 - val_loss: 0.0611 - val_fn: 8.0000 - val_fp: 823.0000 - val_tn: 56063.0000 - val_tp: 67.0000 - val_precision: 0.0753 - val_recall: 0.8933 - 571ms/epoch - 5ms/step
Epoch 6/30
112/112 - 1s - loss: 7.6486e-07 - fn: 18.0000 - fp: 6023.0000 - tn: 221406.0000 - tp: 399.0000 - precision: 0.0621 - recall: 0.9568 - val_loss: 0.0359 - val_fn: 10.0000 - val_fp: 597.0000 - val_tn: 56289.0000 - val_tp: 65.0000 - val_precision: 0.0982 - val_recall: 0.8667 - 587ms/epoch - 5ms/step
Epoch 7/30
112/112 - 1s - loss: 7.5785e-07 - fn: 16.0000 - fp: 7294.0000 - tn: 220135.0000 - tp: 401.0000 - precision: 0.0521 - recall: 0.9616 - val_loss: 0.0436 - val_fn: 8.0000 - val_fp: 738.0000 - val_tn: 56148.0000 - val_tp: 67.0000 - val_precision: 0.0832 - val_recall: 0.8933 - 578ms/epoch - 5ms/step
Epoch 8/30
112/112 - 1s - loss: 7.0495e-07 - fn: 12.0000 - fp: 7115.0000 - tn: 220314.0000 - tp: 405.0000 - precision: 0.0539 - recall: 0.9712 - val_loss: 0.0442 - val_fn: 7.0000 - val_fp: 860.0000 - val_tn: 56026.0000 - val_tp: 68.0000 - val_precision: 0.0733 - val_recall: 0.9067 - 760ms/epoch - 7ms/step
Epoch 9/30
112/112 - 1s - loss: 7.9942e-07 - fn: 13.0000 - fp: 5398.0000 - tn: 222031.0000 - tp: 404.0000 - precision: 0.0696 - recall: 0.9688 - val_loss: 0.1784 - val_fn: 4.0000 - val_fp: 3154.0000 - val_tn: 53732.0000 - val_tp: 71.0000 - val_precision: 0.0220 - val_recall: 0.9467 - 843ms/epoch - 8ms/step
Epoch 10/30
112/112 - 1s - loss: 8.9241e-07 - fn: 12.0000 - fp: 7463.0000 - tn: 219966.0000 - tp: 405.0000 - precision: 0.0515 - recall: 0.9712 - val_loss: 0.0360 - val_fn: 13.0000 - val_fp: 374.0000 - val_tn: 56512.0000 - val_tp: 62.0000 - val_precision: 0.1422 - val_recall: 0.8267 - 823ms/epoch - 7ms/step
Epoch 11/30
112/112 - 1s - loss: 7.3766e-07 - fn: 17.0000 - fp: 6367.0000 - tn: 221062.0000 - tp: 400.0000 - precision: 0.0591 - recall: 0.9592 - val_loss: 0.1282 - val_fn: 3.0000 - val_fp: 3058.0000 - val_tn: 53828.0000 - val_tp: 72.0000 - val_precision: 0.0230 - val_recall: 0.9600 - 726ms/epoch - 6ms/step
Epoch 12/30
112/112 - 1s - loss: 6.2805e-07 - fn: 11.0000 - fp: 6270.0000 - tn: 221159.0000 - tp: 406.0000 - precision: 0.0608 - recall: 0.9736 - val_loss: 0.0516 - val_fn: 7.0000 - val_fp: 995.0000 - val_tn: 55891.0000 - val_tp: 68.0000 - val_precision: 0.0640 - val_recall: 0.9067 - 655ms/epoch - 6ms/step
Epoch 13/30
112/112 - 1s - loss: 7.1583e-07 - fn: 13.0000 - fp: 6592.0000 - tn: 220837.0000 - tp: 404.0000 - precision: 0.0577 - recall: 0.9688 - val_loss: 0.0791 - val_fn: 6.0000 - val_fp: 1616.0000 - val_tn: 55270.0000 - val_tp: 69.0000 - val_precision: 0.0409 - val_recall: 0.9200 - 563ms/epoch - 5ms/step
Epoch 14/30
112/112 - 1s - loss: 1.9508e-06 - fn: 16.0000 - fp: 9941.0000 - tn: 217488.0000 - tp: 401.0000 - precision: 0.0388 - recall: 0.9616 - val_loss: 0.3863 - val_fn: 10.0000 - val_fp: 984.0000 - val_tn: 55902.0000 - val_tp: 65.0000 - val_precision: 0.0620 - val_recall: 0.8667 - 574ms/epoch - 5ms/step
Epoch 15/30
112/112 - 1s - loss: 2.6999e-06 - fn: 26.0000 - fp: 6562.0000 - tn: 220867.0000 - tp: 391.0000 - precision: 0.0562 - recall: 0.9376 - val_loss: 0.0331 - val_fn: 10.0000 - val_fp: 321.0000 - val_tn: 56565.0000 - val_tp: 65.0000 - val_precision: 0.1684 - val_recall: 0.8667 - 647ms/epoch - 6ms/step
Epoch 16/30
112/112 - 1s - loss: 8.1070e-07 - fn: 13.0000 - fp: 7930.0000 - tn: 219499.0000 - tp: 404.0000 - precision: 0.0485 - recall: 0.9688 - val_loss: 0.0403 - val_fn: 11.0000 - val_fp: 717.0000 - val_tn: 56169.0000 - val_tp: 64.0000 - val_precision: 0.0819 - val_recall: 0.8533 - 553ms/epoch - 5ms/step
Epoch 17/30
112/112 - 1s - loss: 7.2819e-07 - fn: 10.0000 - fp: 6555.0000 - tn: 220874.0000 - tp: 407.0000 - precision: 0.0585 - recall: 0.9760 - val_loss: 0.0524 - val_fn: 11.0000 - val_fp: 421.0000 - val_tn: 56465.0000 - val_tp: 64.0000 - val_precision: 0.1320 - val_recall: 0.8533 - 644ms/epoch - 6ms/step
Epoch 18/30
112/112 - 1s - loss: 8.6263e-07 - fn: 6.0000 - fp: 5081.0000 - tn: 222348.0000 - tp: 411.0000 - precision: 0.0748 - recall: 0.9856 - val_loss: 0.0668 - val_fn: 9.0000 - val_fp: 856.0000 - val_tn: 56030.0000 - val_tp: 66.0000 - val_precision: 0.0716 - val_recall: 0.8800 - 559ms/epoch - 5ms/step
Epoch 19/30
112/112 - 1s - loss: 5.2126e-07 - fn: 6.0000 - fp: 4527.0000 - tn: 222902.0000 - tp: 411.0000 - precision: 0.0832 - recall: 0.9856 - val_loss: 0.0219 - val_fn: 11.0000 - val_fp: 457.0000 - val_tn: 56429.0000 - val_tp: 64.0000 - val_precision: 0.1228 - val_recall: 0.8533 - 565ms/epoch - 5ms/step
Epoch 20/30
112/112 - 1s - loss: 3.8198e-07 - fn: 5.0000 - fp: 3450.0000 - tn: 223979.0000 - tp: 412.0000 - precision: 0.1067 - recall: 0.9880 - val_loss: 0.0137 - val_fn: 14.0000 - val_fp: 247.0000 - val_tn: 56639.0000 - val_tp: 61.0000 - val_precision: 0.1981 - val_recall: 0.8133 - 567ms/epoch - 5ms/step
Epoch 21/30
112/112 - 1s - loss: 4.6827e-07 - fn: 6.0000 - fp: 4803.0000 - tn: 222626.0000 - tp: 411.0000 - precision: 0.0788 - recall: 0.9856 - val_loss: 0.0174 - val_fn: 9.0000 - val_fp: 311.0000 - val_tn: 56575.0000 - val_tp: 66.0000 - val_precision: 0.1751 - val_recall: 0.8800 - 625ms/epoch - 6ms/step
Epoch 22/30
112/112 - 1s - loss: 2.5716e-07 - fn: 3.0000 - fp: 2840.0000 - tn: 224589.0000 - tp: 414.0000 - precision: 0.1272 - recall: 0.9928 - val_loss: 0.0232 - val_fn: 12.0000 - val_fp: 487.0000 - val_tn: 56399.0000 - val_tp: 63.0000 - val_precision: 0.1145 - val_recall: 0.8400 - 662ms/epoch - 6ms/step
Epoch 23/30
112/112 - 1s - loss: 3.1180e-07 - fn: 4.0000 - fp: 3034.0000 - tn: 224395.0000 - tp: 413.0000 - precision: 0.1198 - recall: 0.9904 - val_loss: 0.0499 - val_fn: 9.0000 - val_fp: 1154.0000 - val_tn: 55732.0000 - val_tp: 66.0000 - val_precision: 0.0541 - val_recall: 0.8800 - 560ms/epoch - 5ms/step
Epoch 24/30
112/112 - 1s - loss: 2.3617e-07 - fn: 0.0000e+00 - fp: 2943.0000 - tn: 224486.0000 - tp: 417.0000 - precision: 0.1241 - recall: 1.0000 - val_loss: 0.0218 - val_fn: 12.0000 - val_fp: 246.0000 - val_tn: 56640.0000 - val_tp: 63.0000 - val_precision: 0.2039 - val_recall: 0.8400 - 578ms/epoch - 5ms/step
Epoch 25/30
112/112 - 1s - loss: 2.2084e-07 - fn: 3.0000 - fp: 2158.0000 - tn: 225271.0000 - tp: 414.0000 - precision: 0.1610 - recall: 0.9928 - val_loss: 0.0150 - val_fn: 11.0000 - val_fp: 284.0000 - val_tn: 56602.0000 - val_tp: 64.0000 - val_precision: 0.1839 - val_recall: 0.8533 - 548ms/epoch - 5ms/step
Epoch 26/30
112/112 - 1s - loss: 2.6131e-07 - fn: 1.0000 - fp: 2912.0000 - tn: 224517.0000 - tp: 416.0000 - precision: 0.1250 - recall: 0.9976 - val_loss: 0.0137 - val_fn: 14.0000 - val_fp: 244.0000 - val_tn: 56642.0000 - val_tp: 61.0000 - val_precision: 0.2000 - val_recall: 0.8133 - 660ms/epoch - 6ms/step
Epoch 27/30
112/112 - 1s - loss: 1.6721e-07 - fn: 1.0000 - fp: 1679.0000 - tn: 225750.0000 - tp: 416.0000 - precision: 0.1986 - recall: 0.9976 - val_loss: 0.0139 - val_fn: 14.0000 - val_fp: 213.0000 - val_tn: 56673.0000 - val_tp: 61.0000 - val_precision: 0.2226 - val_recall: 0.8133 - 556ms/epoch - 5ms/step
Epoch 28/30
112/112 - 1s - loss: 4.0277e-07 - fn: 4.0000 - fp: 3666.0000 - tn: 223763.0000 - tp: 413.0000 - precision: 0.1013 - recall: 0.9904 - val_loss: 0.0170 - val_fn: 12.0000 - val_fp: 328.0000 - val_tn: 56558.0000 - val_tp: 63.0000 - val_precision: 0.1611 - val_recall: 0.8400 - 751ms/epoch - 7ms/step
Epoch 29/30
112/112 - 1s - loss: 2.7137e-07 - fn: 1.0000 - fp: 3406.0000 - tn: 224023.0000 - tp: 416.0000 - precision: 0.1088 - recall: 0.9976 - val_loss: 0.0117 - val_fn: 13.0000 - val_fp: 218.0000 - val_tn: 56668.0000 - val_tp: 62.0000 - val_precision: 0.2214 - val_recall: 0.8267 - 809ms/epoch - 7ms/step
Epoch 30/30
112/112 - 1s - loss: 3.3463e-07 - fn: 5.0000 - fp: 3602.0000 - tn: 223827.0000 - tp: 412.0000 - precision: 0.1026 - recall: 0.9880 - val_loss: 0.0171 - val_fn: 10.0000 - val_fp: 322.0000 - val_tn: 56564.0000 - val_tp: 65.0000 - val_precision: 0.1680 - val_recall: 0.8667 - 781ms/epoch - 7ms/step
<keras.src.callbacks.History at 0x782b95ee2740>
```
# 結論
訓練結束時，在 56,961 筆驗證交易中，我們：

正確識別其中65個為欺詐

缺失 10 筆詐欺交易

以錯誤標記 404 筆合法交易為代價

參考網址:https://keras.io/examples/structured_data/imbalanced_classification/
