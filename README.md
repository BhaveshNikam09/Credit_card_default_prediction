# Credit Card Default Prediction

## Project Overview
This project focuses on predicting whether a customer will default on their credit card payment based on historical transaction and demographic data. Using machine learning techniques, the model helps financial institutions assess risk and make informed lending decisions.

## Dataset
The dataset contains various customer attributes, which influence credit card default likelihood. The key columns include:

- `limit_balance`: Credit limit of the cardholder
- `sex`: Gender of the cardholder
- `education`: Education level
- `marriage`: Marital status
- `age`: Age of the cardholder
- `pay_0` to `pay_6`: Repayment status for the last six months
- `bill_amt1` to `bill_amt6`: Past bill statements for the last six months
- `pay_amt1` to `pay_amt6`: Amount paid in the last six months
- `default_payment_next_month`: Target variable (1: Default, 0: No Default)

## Project Steps

### 1. Data Preprocessing
- **Cleaning**: Handle missing values and inconsistencies.
- **Feature Encoding**: Convert categorical variables (`education`, `marriage`, `sex`) into numerical form.
- **Scaling**: Standardize numerical features for better model performance.

### 2. Exploratory Data Analysis (EDA)
- **Visualization**: Analyze feature distributions and relationships with `default_payment_next_month`.
- **Correlation Analysis**: Identify key predictors of credit default.

### 3. Feature Engineering
- **New Features**: Create derived features if necessary.
- **Feature Selection**: Use correlation analysis and feature importance techniques.

### 4. Model Building
Several machine learning models were evaluated:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **XGBoost**
- random forest is best suitable for it 

The best-performing model based on evaluation metrics was selected for deployment.

### 5. Model Evaluation
- **Metrics Used**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Cross-Validation**: Ensured model stability and generalization.


### 6. Experiment Tracking with MLflow
MLflow is used to track model experiments, logging parameters and performance metrics for comparison.
- Start the MLflow server:
  ```sh
  mlflow ui
  ```

### 7. DVC (Data Version Control) is used for dataset versioning and model training pipeline management.

  1. Initialize DVC:
     ```
     dvc init
     ```

  2. Add the dataset to DVC:
     ```
     dvc add data/credit_card.csv
     ```

  3. Create a pipeline in `dvc.yaml` to automate preprocessing, training, and evaluation.

  4. Run the pipeline:
     ```
     dvc repro
     ```

  5. Track DVC pipeline and dataset changes in Git:
     ```
     git add dvc.yaml data/credit_card.csv.dvc
     git commit -m "Add data and pipeline setup with DVC"
     ```

**Best Model:** Random Forest
**Performance:** Achieved an AUC-ROC score of approximately **0.85**, indicating strong predictive capability.

## Installation
  1. Clone the repository:
     ```
     git clone https://github.com/yourusername/credit-card-default-prediction.git
     ```
  
  2. Navigate to the project directory:
     ```
     cd credit-card-default-prediction
     ```
  
  3. Install dependencies:
     ```
     pip install -r requirements.txt
     ```

## Usage
  1. Run the DVC pipeline:
     ```
     dvc repro
     ```

  2. Run the model training script:
     ```
     python src/credit_card_prediction/pipelines/training_pipeline.py
     ```

  3. Start the web app for predictions:
     ```
     python app.py
     ```

## Contributors
- **Bhavesh Nikam** ([GitHub](https://github.com/BhaveshNikam09))
- **Harshal Patil** ([GitHub](https://github.com/harshal4257))

## License
This project is licensed under the MIT License.

