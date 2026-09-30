import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

df = pd.read_csv('Perfumes_dataset.csv')
df = df[df['brand'] != 'Brand'].dropna(axis=0)

longevity_map = {'Light': 3, 'Medium': 6, 'Strong': 9, 'Very Strong': 12}
df['y'] = df['longevity'].map(longevity_map).fillna(6)

X = df[['brand', 'type', 'category', 'target_audience']]
y = df['y']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

categorical_cols = ['brand', 'type', 'category', 'target_audience']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# 1. Pipeline con Árbol de Decisión Simple
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=1))
])
dt_pipeline.fit(X_train, y_train)
dt_preds = dt_pipeline.predict(X_valid)
print("MAE - Decision Tree:", mean_absolute_error(y_valid, dt_preds))

# 2. Pipeline con Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=1))
])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_valid)
print("MAE - Random Forest:", mean_absolute_error(y_valid, rf_preds))

# 3. Pipeline con XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=1))
])
xgb_pipeline.fit(X_train, y_train)
xgb_preds = xgb_pipeline.predict(X_valid)
print("MAE - XGBoost:", mean_absolute_error(y_valid, xgb_preds))
