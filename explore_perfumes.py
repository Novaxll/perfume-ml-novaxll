import warnings
warnings.filterwarnings('ignore')
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('Perfumes_dataset.csv')
df = df[df['brand'] != 'Brand'].dropna(axis=0)

print("\n" + "="*65)
print(" 1. EXPLORACIÓN DE DATOS ".center(65, "="))
print("="*65)

print("\n┌── Muestra Inicial (Primeras 5 Filas) " + "─"*30)
print(df[['brand', 'perfume', 'type', 'category', 'target_audience', 'longevity']].head().to_string(index=False))

print("\n┌── Resumen Estadístico (Variables Categóricas) " + "─"*20)
print(df.describe(include='object').to_string())

print("\n┌── Columnas Detectadas " + "─"*42)
print(" | ".join(df.columns))
