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

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=0)
prep = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['brand', 'type', 'category', 'target_audience'])])

dt = Pipeline([('prep', prep), ('model', DecisionTreeRegressor(random_state=1))])
dt.fit(X_tr, y_tr)
mae_dt = mean_absolute_error(y_va, dt.predict(X_va))

rf = Pipeline([('prep', prep), ('model', RandomForestRegressor(n_estimators=100, random_state=1))])
rf.fit(X_tr, y_tr)
mae_rf = mean_absolute_error(y_va, rf.predict(X_va))

xgb = Pipeline([('prep', prep), ('model', XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=1))])
xgb.fit(X_tr, y_tr)
mae_xgb = mean_absolute_error(y_va, xgb.predict(X_va))

res = pd.DataFrame({
    'Model': ['DecisionTree', 'RandomForest', 'XGBoost'],
    'MAE_Hours': [round(mae_dt, 4), round(mae_rf, 4), round(mae_xgb, 4)],
    'MAE_Minutes': [int(mae_dt*60), int(mae_rf*60), int(mae_xgb*60)]
})
print(res.to_string(index=False))
