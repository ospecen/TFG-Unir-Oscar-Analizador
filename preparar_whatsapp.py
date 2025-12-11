# Proyecto desarrollado como parte del Trabajo de Fin de Grado
# Grado en Ingeniería Informática de UNIR.
# Autor: Oscar Pérez Centeno
# Año: 2025-2026

import re
import pandas as pd

def limpiar_whatsapp(path_file):
    
    # Formato típico de WhatsApp:
    # 01/12/2025 21:15 - Oscar: Hola que tal
    patron = r"^\d{1,2}/\d{1,2}/\d{2,4}.*?-\s([^:]+):\s(.*)$"

    autores = []
    textos = []

    with open(path_file, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue

            match = re.match(patron, linea)
            if match:
                autor = match.group(1)
                texto = match.group(2)

                autores.append(autor)
                textos.append(texto)

    df = pd.DataFrame({"autor": autores, "texto": textos})
    return df


def main():
    # RUTA DEL ARCHIVO DE WHATSAPP (.txt)
    ruta = input("Introduce la ruta del archivo WhatsApp .txt: ")

    df = limpiar_whatsapp(ruta)
    print("\nPrimeras filas limpiadas:")
    print(df.head())

    # Guardar en CSV
    salida = "whatsapp_clean.csv"
    df.to_csv(salida, index=False, encoding="utf-8")

    print(f"\nArchivo generado: {salida}")
    print("Filas totales:", len(df))


if __name__ == "__main__":
    main()
