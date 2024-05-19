import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv('/Users/habibaalaa/Downloads/OnlineArticlesPopularity_classification_test.csv')
df.columns = df.columns.str.strip()
df = df.rename(columns={'channel type': 'channel_type'})
df = df.rename(columns={'Article Popularity': 'Article_Popularity'})

# Define column names for numerical and categorical features
numerical_features = ['self_reference_avg_sharess', 'LDA_00', 'num_hrefs',
                      'n_unique_tokens', 'kw_max_avg', 'LDA_03', 'timedelta',
                      'LDA_02', 'LDA_01', 'kw_avg_max', 'num_imgs', 'LDA_04',
                      'self_reference_min_shares', 'kw_max_min', 'kw_avg_avg', 'kw_min_avg']

categorical_features = ['channel_type', 'weekday', 'isWeekEnd']

# Split data into features and target variable
X = df[numerical_features + categorical_features]
y = df['Article_Popularity']

# Split data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the model
loaded_model = load('/Users/habibaalaa/Downloads/ML /Models/gb_model_classfication.joblib')

# Predictions
y_pred = loaded_model.predict(X)

# Evaluate model
accuracy = accuracy_score(y, y_pred)*100
print("Accuracy:", accuracy)
