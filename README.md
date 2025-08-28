# 🧠 Parkinson’s Disease Prediction  

## 📌 Project Overview
This project predicts **Parkinson’s Disease** using the UCI dataset of voice features.  
It includes **extended EDA**, **12 different ML models (including deep learning)**, and a **Streamlit app** with a *Promote* button to retrain and update the best model live.

---

## ⚙️ Models Tested
The following models were trained and evaluated (5-fold cross validation, ROC-AUC):  

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Extra Trees  
- AdaBoost  
- KNN (K-Nearest Neighbors)  
- SVC (Support Vector Machine)  
- MLP (Sklearn)  
- XGBoost  
- LightGBM  
- CatBoost  
- Keras Neural Network (Deep Learning)

---

## 🏆 Best Model
In most experiments, the best performing models were:  
**LightGBM** or **CatBoost**, with ROC-AUC ≈ **0.93–0.95**.

---

## 📊 Example Results

| Model               | ROC-AUC |
|----------------------|---------|
| Logistic Regression  | 0.873   |
| Random Forest        | 0.912   |
| Gradient Boosting    | 0.905   |
| Extra Trees          | 0.910   |
| AdaBoost             | 0.902   |
| KNN                  | 0.881   |
| SVC                  | 0.889   |
| MLP (Sklearn)        | 0.895   |
| XGBoost              | 0.920   |
| LightGBM             | 0.927   |
| CatBoost             | 0.931   |
| Keras NN             | 0.918   |

📈 ROC-AUC comparison:  
![Model Comparison](assets/model_comparison.png)

---

## 📊 Confusion Matrix & ROC Curve (Best Model)

Confusion Matrix of the best model:  
![Confusion Matrix](assets/confusion_matrix.png)

ROC Curve of the best model:  
![ROC Curve](assets/roc_curve.png)

---

## 🖥️ Streamlit App
The app provides:
- 📊 Model comparison table + chart  
- 📝 Form for new patient prediction  
- ✅/❌ Clear result with probability  
- ⚡ *Promote Button*: retrains all models and updates the best one live  

Run locally:
```bash
pip install -r requirements.txt
python model_pipeline.py
streamlit run streamlit_app.py
```

Or in Colab:
```python
!pip install -r requirements.txt
!python model_pipeline.py
!streamlit run streamlit_app.py & npx localtunnel --port 8501
```

---

## 📂 Folder Structure
```
parkinsons_final/
│── data/parkinsons.csv
│── models/best_model.joblib
│── assets/*.png
│── eda_analysis.py
│── model_pipeline.py
│── streamlit_app.py
│── parkinsons_full.ipynb
│── requirements.txt
│── README.md
```
