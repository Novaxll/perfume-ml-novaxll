import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('Perfumes_dataset.csv')
df = df[df['brand'] != 'Brand'].dropna(axis=0)

longevity_map = {'Light': 3, 'Medium': 6, 'Strong': 9, 'Very Strong': 12}
df['y'] = df['longevity'].map(longevity_map).fillna(6)

X = pd.get_dummies(df[['brand', 'type', 'category', 'target_audience']], drop_first=True)
y = df['y']

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)

print("\n" + "="*65)
print(" 2. ENTRENAMIENTO DEL PRIMER MODELO ".center(65, "="))
print("="*65)

print(f"\n• Matriz de características X : {X.shape[0]} registros × {X.shape[1]} columnas")
print(f"• Estado de entrenamiento    : Completado exitosamente")

preds = model.predict(X.head())
comparison = pd.DataFrame({
    'Perfume': df['perfume'].head(),
    'Duración Real (h)': y.head().values,
    'Predicción (h)': preds
})

print("\n┌── Comparativa de Muestra Inicial " + "─"*32)
print(comparison.to_string(index=False))
