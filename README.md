## 🧠 MSCI 因子擁擠模型簡介

MSCI 的「綜合因子擁擠模型」（Integrated Factor Crowding Model）旨在量化評估因子策略的擁擠程度，協助投資人辨識市場中可能過度擁擠的風險因子。該模型融合以下五大維度的指標，並整合為單一標準化的擁擠分數：

- **估值價差 (Valuation Spread)**  
  測量因子高曝險與低曝險標的之估值差距  
- **放空利率差 (Short Interest Spread)**  
  衡量不同曝險群組間的平均放空比率差異  
- **成對相關性 (Pairwise Correlation)**  
  因子頂/底部五分位標的間的超額報酬相關性  
- **因子波動度 (Factor Volatility)**  
  因子預期波動度相對於整體市場的比值  
- **因子長期反轉 (Factor Reversal)**  
  以過去 36 個月累積報酬衡量長期反轉訊號

---

## 📊 使用方式

請先準備好一個含有 MultiIndex（時間、股票代碼）且 column 為因子名稱的 Panel Data。

```python
# 執行 crowding 模型主程式
python get_crowding_model.py

input_path = /Users/yangzherui/Desktop/py coding/因子研究/data/cmoney_eqlw_twse.pkl
output_path = /
factor_return_type = quantile_ls  # 或 ic
