# 🧠 Parkinson’s Disease Prediction  

## 📌 Project Description
This project predicts **Parkinson’s Disease** based on voice features (UCI dataset).  
Multiple ML models are trained and compared (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, MLP).  
The best model is saved and used in a **Streamlit app** with an interactive UI.

---

## 📊 Folder Structure
```
parkinsons_final/
│── data/
│   └── parkinsons.csv
│
│── models/
│   └── best_model.joblib
│
│── assets/
│   ├── hero.png
│   ├── roc_readme.png
│   ├── correlation_heatmap.png
│   ├── target_distribution.png
│   ├── pca_projection.png
│   └── ... (boxplots)
│
│── config.py
│── model_pipeline.py
│── streamlit_app.py
│── eda_analysis.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Run

### 🔹 Local Run
```bash
pip install -r requirements.txt
python eda_analysis.py
python model_pipeline.py
streamlit run streamlit_app.py
```

### 🔹 Colab Run
```python
!pip install -r requirements.txt
!python eda_analysis.py
!python model_pipeline.py
!streamlit run streamlit_app.py & npx localtunnel --port 8501
```

---

## 📈 Results (example)
| Model               | ROC-AUC |
|----------------------|---------|
| Logistic Regression  | 0.873   |
| Random Forest        | 0.912   |
| Gradient Boosting    | 0.905   |
| MLP (Neural Net)     | 0.895   |
| XGBoost              | 0.920   |
| **LightGBM**         | **0.927** |

---

## 🖥️ Streamlit App
- Compare models (table + bar chart).  
- Input new patient features.  
- Clear result: ✅ Healthy / ❌ Parkinson’s with probability.  
- **Promote Button**: retrains all models, updates best one automatically.

---
