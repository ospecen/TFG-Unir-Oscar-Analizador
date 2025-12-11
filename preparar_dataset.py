import pandas as pd

INPUT = "D:\OneDrive\Documentos\Adaptación al Grado en Ingeniería Informática\GRADO INFORMATICA\TFG\MANUEL\dataset\es_hf_102024.csv"
OUTPUT = "D:\OneDrive\Documentos\Adaptación al Grado en Ingeniería Informática\GRADO INFORMATICA\TFG\MANUEL\dataset\dataset_clean.csv"

print("Cargando CSV original...")
df = pd.read_csv(
    INPUT,
    engine="python",
    on_bad_lines="skip"   # evita errores por líneas raras
)

print("Columnas encontradas:", df.columns)

# Renombrar columna
df = df.rename(columns={"labels": "label"})

# Eliminar filas sin texto
df = df.dropna(subset=["text", "label"])

# Convertir etiquetas a enteros
df["label"] = df["label"].astype(float).astype(int)

# Guardar solo text + label
df = df[["text", "label"]]

df.to_csv(OUTPUT, index=False)

print("Dataset limpio guardado en:", OUTPUT)
print(df.head())
print(df["label"].value_counts())