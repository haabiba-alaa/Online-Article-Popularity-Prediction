import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load

# Load your dataset
df = pd.read_csv('/Users/habibaalaa/Downloads/OnlineArticlesPopularity_test.csv')
df.columns = df.columns.str.strip()
df = df.rename(columns={'channel type': 'channel_type'})

# Define column names for numerical and categorical features
numerical_features = ['self_reference_max_shares', 'kw_max_avg', 'LDA_00', 'LDA_03',
                      'rate_positive_words', 'LDA_02', 'kw_max_min',
                      'self_reference_min_shares', 'kw_avg_avg', 'num_videos',
                      'self_reference_avg_sharess', 'LDA_01', 'kw_min_avg', 'kw_avg_max']

categorical_features = ['channel_type', 'weekday', 'isWeekEnd']

X = df[numerical_features + categorical_features]
y = df['shares']

scaler = load('/Users/habibaalaa/Downloads/ML /Models/scaler_shares.joblib')
y_standardized = scaler.transform(y.values.reshape(-1, 1))

#X_train, X_test, y_train, y_test = train_test_split(X, y_standardized, test_size=0.2, random_state=42)

# Load the model
loaded_model = load('/Users/habibaalaa/Downloads/ML /Models/random_forest_model.joblib')

# Use the loaded model to make predictions
y_pred_test_loaded = loaded_model.predict(X)

# Calculate evaluation metrics for the loaded model
mse_test_loaded = mean_squared_error(y_standardized, y_pred_test_loaded)
r2_test_loaded = r2_score(y_standardized, y_pred_test_loaded) * 100

print(f"MSE: {mse_test_loaded}")
print(f"R^2 Score: {r2_test_loaded}")
