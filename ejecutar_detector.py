# Proyecto desarrollado como parte del Trabajo de Fin de Grado
# Grado en Ingeniería Informática de UNIR.
# Autor: Oscar Pérez Centeno
# Año: 2025-2026

# IMPORTAMOS BIBLIOTECAS
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



# 5.3 PREPROCESAMIENTO BÁSICO
# ------------------------------------------------------------------------

def clean_text(text: str) -> str:
    # Pasar a minúsculas y quitar URLs y menciones
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)   # Quitar posibles URLs
    text = re.sub(r"@\w+", "", text)              # Eliminar menciones
    return text.strip()

#Realizar la clasificación
def interpretar_proba(proba_ilicito: float) -> str:
    if proba_ilicito >= 0.65:
        return "Posible discurso ilícito"
    elif proba_ilicito <= 0.45:
        return "Texto no ilícito"
    else:
        return "Es necesaria la revisión humana"


# CLASIFICAR CSV DE WHATSAPP
# ---------------------------------------------------------

def clasificar_whatsapp(csv_path, vectorizer, clf):
    try:
        df = pd.read_csv(csv_path)
    except:
        print("ERROR al leer el archivo. ¿Ruta correcta?")
        return

    if "texto" not in df.columns:
        print("El CSV debe tener una columna llamada: texto")
        return

    # Limpiar texto igual que en el resto del proyecto
    df["clean"] = df["texto"].apply(clean_text)

    # Vectorizar
    X = vectorizer.transform(df["clean"])

    # Probabilidades de clase 1 (posible ilícito)
    probas = clf.predict_proba(X)[:, 1]
    df["prob_ilicito"] = probas

    # Interpretar según umbrales
    df["decision"] = df["prob_ilicito"].apply(interpretar_proba)

    print("\n RESULTADOS DEL ANALISIS DE WHATSAPP")
    # Mostrar algunas filas por pantalla
    for _, fila in df.head(50).iterrows():  # 50 primeros para no llenar la consola
        print("Texto:")
        print(f"  {fila['texto']}")
        print(f"Prob. ilícito: {fila['prob_ilicito']:.2f}")
        print(f"Decisión: {fila['decision']}")
        print("-" * 60)

    # Guardar todo en CSV
    df.to_csv("whatsapp_clasificado.csv", index=False, encoding="utf-8")
    print("\n Archivo exportado: whatsapp_clasificado.csv\n")

# PROGRAMA PRINCIPAL
#-----------------------------------------------------------------------

def main():
    # 1: Cargar el dataset limpio
    print("Cargando dataset limpio...")
    df = pd.read_csv(
        r"D:\OneDrive\Documentos\Adaptación al Grado en Ingeniería Informática\GRADO INFORMATICA\TFG\MANUEL\dataset\dataset_clean.csv",
        engine="python"
    )

    print("Filas totales:", len(df))
    print("Primeras filas:")
    print(df.head())

    # 2: Aplicar preprocesamiento básico al texto
    print("\nAplicando limpieza básica al texto...")
    df["clean_text"] = df["text"].apply(clean_text)

    # 3: Definir X (entrada) e y (etiquetas)
    X = df["clean_text"]
    y = df["label"]

    
    # 5.6 DIVISIÓN DATASET
    #-------------------------------------------------------------
    #Dividir datos
    print("\nDividiendo en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=66,
        stratify=y
    )

    
    # 5.4 REPRESENTACIÓN Bag-of-Words (Vectorización)
    #-----------------------------------------------------------------------
    print("\nVectorizando el texto con Bag-of-Words...")
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    
    # 5.5 MODELO DE CLASIFICACIÓN (Entrenamiento)
    #--------------------------------------------------------------------------
    print("\nEntrenando modelo de Regresión Logística...")
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    
    # 5.7 EVALUACIÓN
    #-------------------------------------------------------------------------
    print("\nEvaluando el modelo en el conjunto de prueba...")
    y_pred = clf.predict(X_test_vec)
    print("\n----INFORME DE CLASIFICACIÓN----")
    print(classification_report(y_test, y_pred))


    #5.7.5 MENU PRINCIPAL
    #------------------------------------------------
    while True:
        print("----MENÚ DEL DETECTOR----")
        print("1 - Modo interactivo")
        print("2 - Analizar archivo de WhatsApp (.csv)")
        print("3 - Salir")

        opcion = input("Selecciona una opción: ")


        if opcion == "1":
            print("\nModo interactivo: Escribe 'salir' para terminar.\n")
            while True:
                user_text = input("Texto: ")
                if user_text.lower().strip() == "salir":
                    break

                user_clean = clean_text(user_text)
                user_vec = vectorizer.transform([user_clean])
                proba_ilicito = clf.predict_proba(user_vec)[0][1]

                # Usamos la función común interpretar_proba
                decision = interpretar_proba(proba_ilicito)

                print(f"{decision}  (prob = {proba_ilicito:.2f})\n")
        
        elif opcion == "2":# 5.8.5 ENTRADA DE WHATSAPP
            ruta = input("\nIntroduce la ruta del archivo CSV de WhatsApp: ")
            clasificar_whatsapp(ruta, vectorizer, clf)

        elif opcion == "3":
            print("¡Hasta pronto!")
            break

        else:
            print("Opción no válida, intentaló de nuevo.\n")

if __name__ == "__main__":
    main()