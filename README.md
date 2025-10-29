Telecom Customer Churn Prediction
This project focuses on building a machine learning model to predict customer churn in the telecom industry. The primary goal is to identify customers at risk of leaving the service, allowing the business to implement targeted retention strategies.

📌 Business Understanding
Customer churn refers to the loss of customers over a given period. In the highly competitive telecom industry, retaining existing customers is significantly more cost-effective than acquiring new ones. High churn rates can negatively impact profitability and growth.

Key challenges in the telecom industry include:

Intense competition and market saturation.

Customer price sensitivity.

Dissatisfaction with service quality or pricing.

The importance of reducing churn:

Improved profitability: Retaining customers reduces high acquisition costs.

Enhanced loyalty: Focusing on retention builds stronger, long-term customer relationships.

Optimized marketing: Enables better targeting of promotions to loyal customers.

🚀 Methodology
The project follows a standard machine learning workflow:

Data Loading & Preprocessing
The data is loaded from two separate CSV files (churn-bigml-20.csv and churn-bigml-80.csv).

The two datasets are concatenated into a single DataFrame for analysis and modeling.

Initial data exploration is performed to check for missing values, duplicates, and data types.

Exploratory Data Analysis (EDA)
The notebook performs a statistical analysis on the numerical features using df.describe().

It separates categorical and numerical columns to prepare for plotting.

Univariate analysis is performed to visualize the distribution of each feature. This helps in understanding the data and identifying potential outliers.

Model Training
Three powerful gradient boosting models—XGBoost, LightGBM, and CatBoost—are used to predict customer churn.

Optuna, an open-source hyperparameter optimization framework, is used to find the best hyperparameters for each model, maximizing predictive performance.

📈 Results and Interpretation
After training and hyperparameter tuning, the models are evaluated using several key metrics:

Accuracy Score: The percentage of correctly predicted churn and non-churn cases.

Confusion Matrix: A detailed breakdown of correct and incorrect predictions, showing True Positives, True Negatives, False Positives, and False Negatives.

The code also uses SHAP (SHapley Additive exPlanations) to interpret the model's predictions and understand which features are most influential in predicting churn.

Summary plots illustrate how features impact predictions across the dataset.

Bar plots show the overall feature importance, providing insights into which factors are the most significant drivers of customer churn.

SHAP values highlight that Total day charge, Customer service calls, and International plan are among the most influential features.

🛠 Getting Started
Clone the repository.

Install the required libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, catboost, and optuna.

Run the Jupyter Notebook: Execute the cells in order to load the data, train the models, and visualize the results.
