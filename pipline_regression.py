import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from joblib import dump

# Load your dataset
df = pd.read_csv('/content/OnlineArticlesPopularity.csv')
df.columns = df.columns.str.strip()
df = df.rename(columns={'channel type': 'channel_type'})

# Define column names for numerical and categorical features
numerical_features = ['self_reference_max_shares', 'kw_max_avg', 'LDA_00', 'LDA_03',
                      'rate_positive_words', 'LDA_02', 'kw_max_min',
                      'self_reference_min_shares', 'kw_avg_avg', 'num_videos',
                      'self_reference_avg_sharess', 'LDA_01', 'kw_min_avg', 'kw_avg_max']

categorical_features = ['channel_type', 'weekday', 'isWeekEnd']

# Define transformations for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Standardize features
])

# Define transformations for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with mode
    ('encoder', OneHotEncoder(drop='first'))  # One-hot encode categorical features
])

# Combine transformers for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Assuming 'df' is your DataFrame and 'shares' is the column you want to standardize
shares_data = df[['shares']]

# Initialize StandardScaler
scaler = StandardScaler()

# Fit scaler to the data and transform
df['standardized_shares'] = scaler.fit_transform(shares_data)

# Define the final pipeline including preprocessing, model, and target variable transformation
pipeline_rf = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', TransformedTargetRegressor(
        regressor=RandomForestRegressor(n_estimators=400, max_depth=3, random_state=42),
        transformer=scaler  # Use the same StandardScaler for target variable transformation
    ))  # Random Forest model with target variable transformation
])

# Split data into features and target variable
X = df[numerical_features + categorical_features]
y = df['standardized_shares']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline_rf.fit(X_train, y_train)

# Save the model
dump(pipeline_rf, 'random_forest_model.joblib')

# Assuming 'y_train' is your training target variable
scaler = StandardScaler()
scaler.fit(y.values.reshape(-1, 1))  # Reshape to ensure it's a column vector

# Save the scaler object
dump(scaler, 'scaler_shares.joblib')
