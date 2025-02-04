# Credit Card Default Prediction

This project aims to predict whether a customer will default on their credit card payment using machine learning techniques. It leverages customer information such as demographics, credit history, and other financial features to predict the likelihood of default. The model is trained using a dataset and evaluated using different machine learning models.

## Project Features
- **Data Preprocessing**: Clean and prepare the dataset for model training.
- **Exploratory Data Analysis (EDA)**: Analyze and visualize the data to understand trends and relationships between features.
- **Feature Engineering**: Select and create features that improve model performance.
- **Modeling**: Train different machine learning models (Logistic Regression, Random Forest, XGBoost, etc.) for prediction.
- **Model Evaluation**: Evaluate the models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
- **Hyperparameter Tuning**: Optimize the models to improve performance using GridSearchCV or RandomizedSearchCV and mainly the Optuna.
- **Model Deployment**: Optionally, deploy the model using a web framework like Flask or FastAPI for real-time predictions.

## Technologies Used
- **Python**
- **Pandas** (Data manipulation)
- **NumPy** (Numerical operations)
- **Matplotlib** and **Seaborn** (Data visualization)
- **Scikit-learn** (Machine learning)
- **XGBoost** (For boosted decision trees)

## Dataset
The dataset used in this project is typically from sources like Kaggle or UCI Machine Learning Repository. The dataset contains information on the following columns:
- **ID**: Unique identifier for the customer.
- **LIMIT_BAL**: Credit limit.
- **SEX**: Gender of the customer.
- **EDUCATION**: Customer's education level.
- **MARRIAGE**: Marital status.
- **AGE**: Age of the customer.
- **PAY_0, PAY_2, ..., PAY_6**: Payment status in the last 6 months (e.g., 1 = payment on time, -1 = payment delayed).
- **BILL_AMT1, BILL_AMT2, ..., BILL_AMT6**: Amount of bill statement for the last 6 months.
- **PAY_AMT1, PAY_AMT2, ..., PAY_AMT6**: Amount paid in the last 6 months.
- **DEFAULT**: Target variable (1 = default, 0 = no default).

## Steps to Run the Project

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/credit-card-default-prediction.git
    cd credit-card-default-prediction
    ```

2. **Install the Required Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebooks**
    Open the Jupyter notebook to explore the data, build models, and evaluate performance.
    ```bash
    jupyter notebook
    ```

4. **Run the Python Script**
    Alternatively, you can run the Python script to train and evaluate the models:
    ```bash
    python credit_card_default_prediction.py
    ```

5. **Optional: Deploy the Model**
    If you want to deploy the model as a web application, you can use **Flask** or **FastAPI**. Instructions for deployment can be found in the `deployment/` directory.

## Model Evaluation

- **Accuracy**: Percentage of correct predictions.
- **Precision**: Fraction of relevant instances among the retrieved instances.
- **Recall**: Fraction of relevant instances that have been retrieved.
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC-AUC**: Measures the area under the Receiver Operating Characteristic curve.

## Results
- The best performing model and its evaluation metrics will be displayed after training.
- **Confusion Matrix**: To visualize the performance of classification models.

## Conclusion
This project demonstrates the use of machine learning models for predicting credit card defaults, which can be helpful for financial institutions in assessing the risk of their customers. The models can be further improved with more sophisticated feature engineering, advanced models like **XGBoost**, or **Deep Learning** models.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The dataset used is from the **UCI Machine Learning Repository** or **Kaggle**.
- Special thanks to the contributors and the open-source community for providing the libraries used in this project.

