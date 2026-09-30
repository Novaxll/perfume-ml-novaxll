import pandas as pd

df = pd.read_csv('Perfumes_dataset.csv')
df = df[df['brand'] != 'Brand'].dropna(axis=0)

print("--- Primeras Filas ---")
print(df.head())

print("\n--- Resumen Estadistico de Categóricas ---")
print(df.describe(include='object'))

print("\n--- Columnas del Dataset ---")
print(df.columns)
