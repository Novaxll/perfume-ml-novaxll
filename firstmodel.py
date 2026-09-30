import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('Perfumes_dataset.csv')
df = df[df['brand'] != 'Brand'].dropna(axis=0)

longevity_map = {'Light': 3, 'Medium': 6, 'Strong': 9, 'Very Strong': 12}
df['y'] = df['longevity'].map(longevity_map).fillna(6)

df_encoded = pd.get_dummies(df[['brand', 'type', 'category', 'target_audience']], drop_first=True)

X = df_encoded
y = df['y']

print("Filas y columnas en X:", X.shape)
print("Target (y) muestreo:")
print(y.head())

model = DecisionTreeRegressor(random_state=1)
print("Entrenando modelo...")
model.fit(X, y)
print("¡Entrenamiento completado!")

print("\nPredicciones para los primeros 5 perfumes:")
print(model.predict(X.head()))
print("Duraciones reales (horas aproximadas):")
print(y.head().values)
