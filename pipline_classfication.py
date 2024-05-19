import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

# Load your dataset
df = pd.read_csv('/content/OnlineArticlesPopularity_Milestone2.csv')
df.columns = df.columns.str.strip()
df = df.rename(columns={'channel type': 'channel_type'})
df = df.rename(columns={'Article Popularity': 'Article_Popularity'})

# Define column names for numerical and categorical features
numerical_features = ['self_reference_avg_sharess', 'LDA_00', 'num_hrefs',
                      'n_unique_tokens', 'kw_max_avg', 'LDA_03', 'timedelta',
                      'LDA_02', 'LDA_01', 'kw_avg_max', 'num_imgs', 'LDA_04',
                      'self_reference_min_shares', 'kw_max_min', 'kw_avg_avg', 'kw_min_avg']

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

# Define the final pipeline including preprocessing and model
pipeline_gb = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
])

# Define target variable for classification
y = df['Article_Popularity']

# Split data into features and target variable
X = df[numerical_features + categorical_features]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline_gb.fit(X_train, y_train)

# Save the model
dump(pipeline_gb, 'gb_model_classfication.joblib')
