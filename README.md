醫美診所客戶辨識系統：基於 Siamese Network 的 One-Shot Learning 實作
Clinic Face Recognition via Siamese Network & One-Shot Learning
📌 專案背景 (Project Overview)
在醫美診所營運中，識別客戶身份是提供客製化服務的第一步。然而，新客戶的臉部影像樣本通常只有 1-2 張，傳統的大型分類模型（Classification）難以在樣本極少的情況下運作。

本專案利用 Siamese Network (孿生網路) 成功實作了 One-Shot Learning：

解決方案：不對人臉進行「分類」，而是學習如何「比較」兩張臉的相似度。

應用場景：僅需每位諮詢師或客戶的一張底圖，系統即可即時判斷監視畫面中的人物身份。

🚀 技術核心 (Technical Core)
1. Siamese Network 架構
特徵提取 (Embedding)：使用兩個共享權重的 CNN 分支，將人臉影像轉換為 128 維的特徵向量。

距離運算 (Distance Layer)：透過計算兩組向量之間的 L1 Distance (Manhattan Distance) 來判斷相似程度。

損失函數 (Loss Function)：採用 Contrastive Loss，確保同一個人的向量距離最小化，不同人則最大化。

2. 影像預處理流程 (Image Pipeline)
使用 OpenCV 實作自動化預處理：

Face Detection：自動定位人臉位置。

Alignment & Resizing：校正臉部角度並統一縮放至模型輸入尺寸（如 100x100）。

Normalization：將像素值歸一化以加速模型收斂。

🛠 核心功能 (Key Features)
One-Shot Learning：解決診所新客戶樣本不足的問題，達到「見一次就能認得」。

即時比對 (Real-time Inference)：優化 Python 程式碼邏輯，實現流暢的即時辨識速度。

高度彈性：當有新員工入職時，無需重新訓練模型，只需增加一張基準照片即可完成註冊。
