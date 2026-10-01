import warnings
warnings.filterwarnings('ignore')
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

dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeRegressor(random_state=1))])
dt_pipeline.fit(X_train, y_train)
mae_dt = mean_absolute_error(y_valid, dt_pipeline.predict(X_valid))

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=1))])
rf_pipeline.fit(X_train, y_train)
mae_rf = mean_absolute_error(y_valid, rf_pipeline.predict(X_valid))

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=1))])
xgb_pipeline.fit(X_train, y_train)
mae_xgb = mean_absolute_error(y_valid, xgb_pipeline.predict(X_valid))

print("\n" + "="*65)
print(" 5. COMPARATIVA DE MODELOS AVANZADOS (PIPELINES) ".center(65, "="))
print("="*65)

res = pd.DataFrame({
    'Algoritmo': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'MAE (Horas)': [f"{mae_dt:.4f}", f"{mae_rf:.4f}", f"{mae_xgb:.4f}"],
    'Margen de Error': [f"~{int(mae_dt*60)} mins", f"~{int(mae_rf*60)} mins", f"~{int(mae_xgb*60)} mins"]
})

print("\n" + res.to_string(index=False) + "\n")
