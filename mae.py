import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('Perfumes_dataset.csv')
df = df[df['brand'] != 'Brand'].dropna(axis=0)

longevity_map = {'Light': 3, 'Medium': 6, 'Strong': 9, 'Very Strong': 12}
df['y'] = df['longevity'].map(longevity_map).fillna(6)

X = pd.get_dummies(df[['brand', 'type', 'category', 'target_audience']], drop_first=True)
y = df['y']

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)

predictions = model.predict(X)
mae_in_sample = mean_absolute_error(y, predictions)

print("\n" + "="*65)
print(" 3. EVALUACIÓN MAE IN-SAMPLE ".center(65, "="))
print("="*65)

print(f"\n• Error Absoluto Medio (In-Sample) : {mae_in_sample:.4f} horas (~{int(mae_in_sample*60)} mins)")
