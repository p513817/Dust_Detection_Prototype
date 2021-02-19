# Dust_Detection_Prototype
Detecting dust with opencv algorithm and deep learning

| 作者 | Chia-Chun, Chang |
| ---- | ---|
| 所屬單位  | Cavedu 教育團隊 |
| 開發日期  | 11001 |

### ROAD CONDITION

![image](figures/dust_detection_demo_gif.gif)

# 粉塵偵測 ( 原型 )

### 前言
此為快速開發的模擬專案
目的是模擬工廠機具的粉塵偵測
**展示專案的技術概念**

### 介紹
透過 AI 影像辨識 加上 CV 影像演算法，將能偵測特定區域的粉塵狀況 \n
若偵測到粉塵則發出警告因並開啟噴嘴進行清掃

### 軟體技術
原先希望僅利用 AI 來完成影像辨識
但因為粉塵為不規則形狀，蒐集資料時怕無法涵蓋到所有狀況
故結合 OpenCV 的演算法進行二次辨識並框出粉塵
利用加權的方式來調整比重

### 硬體
使用樹莓派加上網路攝影機、電磁閥、蜂鳴器即可完成

### 結論
因為只耗時四天完成此原型程式碼較混亂、效能較不佳
請多多擔待
