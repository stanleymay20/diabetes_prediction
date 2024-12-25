# Diabetes Prediction using Machine Learning

## Project Overview
This project aims to predict diabetes status using patient data. The objective is to build a machine learning model that can assist in identifying individuals at risk of diabetes.

## Dataset
The dataset contains the following features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: Serum insulin (μU/mL)
- **BMI**: Body Mass Index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history
- **Age**: Age in years
- **Outcome**: Diabetes diagnosis (1 for positive, 0 for negative)

## Data Preprocessing
- Replaced invalid zero values in features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with the median values.
- Standardized and normalized features for improved model performance.

## Exploratory Data Analysis (EDA)
- Visualized data distributions and correlations.
- Identified feature importance for predicting the `Outcome` variable.
- Full EDA report is available in the `reports/EDA_Diabetes_dataset.html`.

## Machine Learning Model
- **Model Used**: Random Forest Classifier
- **Training Process**:
  - Split the dataset into 80% training and 20% testing sets.
  - Trained the model with default hyperparameters.

## Model Evaluation
- **Accuracy**: 74.68%
- **Precision, Recall, F1-Score**:
  ```
              precision    recall  f1-score   support

           0       0.81      0.79      0.80        99
           1       0.64      0.67      0.65        55

    accuracy                           0.75       154
   macro avg       0.73      0.73      0.73       154
weighted avg       0.75      0.75      0.75       154
  ```
- **Confusion Matrix**:
  ```
  [[78, 21],
   [18, 37]]
  ```

## Repository Structure
```
├── data
│   └── diabetes.csv
├── notebooks
│   └── EDA_and_Modeling.ipynb
├── models
│   └── trained_random_forest.pkl
├── reports
│   └── EDA_Diabetes_dataset.html
├── README.md
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/stanleymay20/diabetes-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd diabetes-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook for training:
   ```bash
   jupyter notebook notebooks/EDA_and_Modeling.ipynb
   ```
5. View the EDA report in `reports/EDA_Diabetes_dataset.html`.

## Next Steps
- Optimize the Random Forest model using hyperparameter tuning.
- Compare performance with other models like Logistic Regression and Gradient Boosting.
- Deploy the model using a web interface for real-time predictions.

## Contributors
- Stanley ([@stanleymay20](https://github.com/stanleymay20))
