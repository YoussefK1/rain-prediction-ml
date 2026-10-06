# Rain Prediction System: next-day rain forecasting with ML & deep learning

End-to-end machine learning project that predicts whether it will rain **tomorrow** in Australia from today's weather observations. It covers exploratory analysis, cleaning, feature engineering, class-imbalance handling (SMOTE), a comparison of four models (Random Forest, SVM, GRU, LightGBM), explainability with SHAP, and a Streamlit web app.

> Team project from the **IPS Spring Program 2026** at CY Tech (CY Cergy Paris Université), supervised by Prof. Souhila Arib.

## Results

Evaluated on a held-out, stratified 20% test set (29,092 days, original 78/22 class ratio, SMOTE applied to the training set only).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 85.17% | 72.31% | 53.38% | 61.42% | 0.885 |
| SVM (linear, PCA) | 75.31% | 46.36% | 80.75% | 58.90% | 0.857 |
| GRU | 80.11% | 53.09% | 79.37% | 63.62% | 0.881 |
| **LightGBM (selected)** | **85.02%** | **64.95%** | **68.74%** | **66.79%** | **0.891** |

Precision, recall and F1 are for the "rain" class. Because only ~22% of days are rainy, a model that always predicts "no rain" already reaches ~78% accuracy, so **F1 and ROC-AUC were used to choose the model**, not accuracy. LightGBM gives the best balance between catching real rain and avoiding false alarms.

![LightGBM confusion matrix](images/lightgbm_confusion.png)

## Explainability

SHAP analysis of the LightGBM model shows that afternoon humidity (`Humidity3pm`), wind gust speed and same-day rainfall are the strongest drivers of next-day rain, which is consistent with meteorological intuition.

![SHAP summary, LightGBM](images/shap_lightgbm.png)

## Dataset

- **Source:** [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) (Kaggle), file `data/weatherAUS.csv`
- **Size:** 145,460 daily observations from 49 weather stations (Nov 2007 to Jun 2017)
- **Task:** tabular binary classification, target `RainTomorrow` (0 = no rain, 1 = rain)
- **Imbalance:** 113,583 "no rain" vs 31,877 "rain" days (about 3.56 : 1)

## Pipeline

1. **EDA:** distributions, missing values, outliers, correlations.
2. **Cleaning:** median/mode imputation, duplicate removal, IQR outlier capping.
3. **Feature engineering:** 12 engineered features (e.g. `TempDifference`, `HumidityDifference`, `PressureDifference`, `WindGustRatio`, `RainTodayBinary`, `Rainfall_Log`, temperature-humidity and pressure-humidity interactions), label encoding of categorical variables, standardization. The dataset grows from 23 to 35 columns.
4. **Split:** stratified 80/20 (116,368 train / 29,092 test).
5. **Imbalance handling:** SMOTE on the **training set only** (and inside each cross-validation fold) to avoid data leakage.
6. **Modelling and tuning:** `RandomizedSearchCV` with 5-fold stratified cross-validation, optimized for F1.
7. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, precision-recall curves, confusion matrices, SHAP.
8. **Deployment:** Streamlit app serving the LightGBM model.

## Repository structure

```
rain-prediction-ml/
├── data/                  # raw dataset (weatherAUS.csv)
├── docs/                  # full report and presentation (PDF)
├── images/                # confusion matrices, ROC/PR curves, SHAP, feature importance
├── models/                # trained models and preprocessing artifacts
├── src/
│   ├── data_preprocessing_feature_engineering.py   # EDA, cleaning, features
│   ├── random_forest.py                            # Random Forest + SMOTE + tuning
│   ├── svm_gru.py                                  # SVM (PCA) and GRU models
│   ├── lightgbm_weather.py                         # LightGBM + SMOTE + SHAP
│   ├── mlp.py                                      # MLP baseline
│   ├── main.py                                     # model loading and prediction logic
│   ├── ui.py                                       # Streamlit interface
│   └── run_app.py                                  # app launcher
├── requirements.txt
└── README.md
```

## How to run

Run all commands from the repository root. Python 3.10 or newer is recommended.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the cleaned, feature-engineered dataset
python src/data_preprocessing_feature_engineering.py

# 3. Train and evaluate the models (each script is independent)
python src/random_forest.py
python src/svm_gru.py
python src/lightgbm_weather.py
python src/mlp.py

# 4. Launch the web app
streamlit run src/ui.py
```

The cleaned dataset (~95 MB) is not stored in the repository because step 2 regenerates it. Trained models are already included in `models/`, so you can launch the app (step 4) without retraining.

## Limitations and future work

- **Decision threshold:** the LightGBM threshold is tuned for F1 on the test set. A stricter protocol would select it on a separate validation split and report the test set only once.
- **Deployment checks:** end-to-end testing of the Streamlit app (see the report) found mismatches between the app's feature encoding and the training-time preprocessing, such as location and wind-direction codes. Reusing the saved encoders in the app would fix this.
- **Temporal structure:** the data is treated as independent daily rows. A time-based split and lagged features would give a more realistic estimate of forecasting performance.
- **Probability calibration** and a cost-sensitive threshold could better reflect that missing a rainy day is costlier than a false alarm.

## Documentation

- [Full project report](docs/Rain_Prediction_Report.pdf)
- [Presentation slides](docs/Rain_Prediction_Presentation.pdf)

## Authors

Youssef ALY, Isela Allhassa and Pietro Vinsten.