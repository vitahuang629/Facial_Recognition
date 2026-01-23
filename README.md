## 專案背景 (Project Overview)
在醫美診所營運中，識別客戶身份是提供客製化服務的第一步。然而，新客戶的臉部影像樣本通常只有 1-2 張，傳統的大型分類模型（Classification）難以在樣本極少的情況下運作。

本專案利用 **Siamese Network (孿生網路)** 成功實作了 **One-Shot Learning**：
* **核心邏輯**：不對人臉進行「分類」，而是學習如何「比較」兩張臉的相似度。
* **解決痛點**：僅需每位諮詢師或客戶的一張基準照片，系統即可在即時影像中判斷人物身份，無需針對新成員重新訓練模型。

---

## 技術架構 (Technical Architecture)

### 1. Siamese Network 模型設計
* **特徵提取 (Embedding)**：使用共享權重的卷積神經網絡 (CNN)，將影像映射至一個 128 維的特徵空間。
* **距離層 (Distance Layer)**：透過計算兩組向量之間的 **L1 Distance (Manhattan Distance)** 來衡量差異。
* **損失函數 (Loss Function)**：採用 **Contrastive Loss**，確保相同身份的特徵距離最小化，不同身份則最大化。



### 2. 影像預處理 Pipeline (OpenCV)
為了提高辨識穩定性，我實作了完整的影像預處理流程：
1. **人臉偵測 (Face Detection)**：使用 OpenCV 偵測影像中的人臉區域。
2. **對齊與縮放 (Alignment & Resizing)**：將人臉裁切並統一縮放至 100x100 像素。
3. **歸一化 (Normalization)**：進行像素值縮放，提升模型推論的收斂速度。

---

## 核心功能 (Key Features)

- **One-Shot Learning**：見一次就能認得，適合客戶流動性大的診所場景。
- **動態註冊**：新增員工或客戶時，只需存入一張照片即可完成部署。
- **即時比對**：優化 Python 程式碼，確保在一般筆電設備上也能達到即時 (Real-time) 辨識。

---

## 測試指南 (Usage Guide)

如果您想要測試本系統的辨識功能，請按照以下步驟操作：

1. **準備驗證照片**：
   - 請將您想要「被核對」的目標人物照片放入 `application_data/verification_images/` 資料夾中。
   - 這些照片將作為系統的「已知身份」資料庫。

2. **執行程式**：
   - 執行 `faceid.py`。
   - 程式啟動後，視窗會顯示來自電腦鏡頭的即時畫面。

3. **進行比對**：
   - 確保鏡頭前有人物（也就是您自己或測試者）。
   - 點擊視窗中的 **"Verify"** 按鈕。
   - 系統會自動捕捉當下的鏡頭畫面 (`input_image`)，並與 `verification_images` 資料夾中的照片進行比對。
   - 比對結果（Verified / Unverified）將顯示在視窗下方。
