"""
modules/filter.py
=================
Filtrado Inteligente de Alimentos

Técnica: Árbol de Decisión (Decision Tree Classifier)
Entrena un árbol para aprender qué alimentos son "aptos" o "no aptos"
según el perfil del usuario. Luego filtra el dataset completo.

Restricciones manejadas:
  - Alergias / restricciones alimentarias (gluten, lactosa, etc.)
  - Límite de calorías por porción
  - Límite de grasa por porción
  - Límite de azúcar por porción (para usuarios diabéticos)
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os


# ─────────────────────────────────────────────
# PALABRAS CLAVE POR RESTRICCIÓN
# Mapea cada restricción a palabras en el nombre del alimento
# ─────────────────────────────────────────────
RESTRICCIONES_KEYWORDS = {
    "gluten":   ["wheat", "flour", "bread", "pasta", "cereal", "barley",
                 "rye", "malt", "biscuit", "cracker", "noodle", "pretzel"],
    "lactosa":  ["milk", "cheese", "butter", "cream", "yogurt", "whey",
                 "dairy", "lactose", "ice cream", "custard"],
    "nueces":   ["nut", "almond", "walnut", "cashew", "peanut", "pecan",
                 "pistachio", "hazelnut", "macadamia"],
    "mariscos": ["shrimp", "crab", "lobster", "clam", "oyster", "scallop",
                 "squid", "octopus", "mussel"],
    "huevo":    ["egg", "mayonnaise", "meringue"],
    "soya":     ["soy", "tofu", "tempeh", "miso", "edamame"],
    "cerdo":    ["pork", "ham", "bacon", "lard", "prosciutto", "salami"],
    "res":      ["beef", "steak", "veal", "bison", "hamburger"],
    "pollo":    ["chicken", "poultry", "turkey"],
    "pescado":  ["fish", "salmon", "tuna", "cod", "tilapia", "sardine",
                 "trout", "halibut", "anchovy"],
    "azucar":   ["sugar", "candy", "chocolate", "syrup", "honey",
                 "jelly", "jam", "dessert", "cake", "cookie", "pie"],
}


# ─────────────────────────────────────────────
# FUNCIONES DE FILTRADO POR REGLAS
# ─────────────────────────────────────────────

def contiene_restriccion(nombre: str, restricciones: list) -> bool:
    """
    Verifica si el nombre de un alimento contiene palabras
    relacionadas con las restricciones del usuario.

    Parámetros:
        nombre        : nombre del alimento (string)
        restricciones : lista de restricciones del usuario

    Retorna:
        True si el alimento DEBE ser excluido
    """
    nombre_lower = nombre.lower()
    for restriccion in restricciones:
        keywords = RESTRICCIONES_KEYWORDS.get(restriccion.lower(), [restriccion.lower()])
        if any(kw in nombre_lower for kw in keywords):
            return True
    return False


def filtrar_por_reglas(df: pd.DataFrame, restricciones: list,
                       max_cal_porcion: float = 800,
                       max_grasa_porcion: float = 50,
                       max_azucar_porcion: float = 40) -> pd.DataFrame:
    """
    Primera pasada: filtra por reglas simples (restricciones y límites).

    Parámetros:
        df               : DataFrame limpio de foods_clean.csv
        restricciones    : lista de restricciones del usuario
        max_cal_porcion  : máximo de calorías permitidas por 100g
        max_grasa_porcion: máximo de grasa permitida por 100g
        max_azucar_porcion: máximo de azúcar permitida por 100g

    Retorna:
        DataFrame filtrado
    """
    df_filtrado = df.copy()

    # Filtrar por restricciones alimentarias
    if restricciones:
        mask = df_filtrado['nombre'].apply(
            lambda x: not contiene_restriccion(x, restricciones)
        )
        antes = len(df_filtrado)
        df_filtrado = df_filtrado[mask]
        print(f"  🚫 Restricciones ({', '.join(restricciones)}): "
              f"{antes - len(df_filtrado)} alimentos eliminados")

    # Filtrar por calorías máximas
    antes = len(df_filtrado)
    df_filtrado = df_filtrado[df_filtrado['calories'] <= max_cal_porcion]
    print(f"  🔥 Calorías > {max_cal_porcion}: {antes - len(df_filtrado)} eliminados")

    # Filtrar por grasa máxima
    antes = len(df_filtrado)
    df_filtrado = df_filtrado[df_filtrado['fat'] <= max_grasa_porcion]
    print(f"  🧈 Grasa > {max_grasa_porcion}g: {antes - len(df_filtrado)} eliminados")

    # Filtrar por azúcar máxima
    antes = len(df_filtrado)
    df_filtrado = df_filtrado[df_filtrado['sugar'] <= max_azucar_porcion]
    print(f"  🍬 Azúcar > {max_azucar_porcion}g: {antes - len(df_filtrado)} eliminados")

    return df_filtrado.reset_index(drop=True)


# ─────────────────────────────────────────────
# ÁRBOL DE DECISIÓN
# ─────────────────────────────────────────────

def generar_etiquetas(df: pd.DataFrame, perfil: dict) -> pd.Series:
    """
    Genera etiquetas automáticas (apto=1 / no apto=0) para entrenar
    el árbol de decisión, basándose en qué tan bien se alinea
    cada alimento con el perfil nutricional del usuario.

    Un alimento es "apto" si:
      - Calorías por 100g <= 60% del objetivo diario / 4 comidas
      - Proteína >= mínima esperada por comida (si la meta es músculo/bajar)
      - Grasa no excede el límite por comida
      - Azúcar no excede el límite

    Parámetros:
        df     : DataFrame filtrado por reglas
        perfil : diccionario del módulo profile.py

    Retorna:
        Serie con etiquetas 0 o 1
    """
    cal_por_comida  = perfil['calorias_objetivo'] / 4
    prot_por_comida = perfil['protein_g'] / 4
    fat_por_comida  = perfil['fat_g'] / 4

    # Una porción estándar es ~150g, ajustamos los valores por 100g
    cal_limite  = (cal_por_comida / 150) * 100
    prot_minima = (prot_por_comida / 150) * 100 * 0.5   # al menos 50% de la proteína mínima
    fat_limite  = (fat_por_comida / 150) * 100 * 1.5    # algo de margen en grasa

    etiquetas = (
        (df['calories'] <= cal_limite) &
        (df['protein']  >= prot_minima) &
        (df['fat']      <= fat_limite)
    ).astype(int)

    return etiquetas


def entrenar_arbol(df_filtrado: pd.DataFrame, perfil: dict) -> tuple:
    """
    Entrena un árbol de decisión para clasificar alimentos
    como aptos (1) o no aptos (0) para el perfil del usuario.

    Parámetros:
        df_filtrado : DataFrame ya filtrado por reglas
        perfil      : diccionario del módulo profile.py

    Retorna:
        (modelo entrenado, accuracy en test, reporte)
    """
    features = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar']
    X = df_filtrado[features].fillna(0)
    y = generar_etiquetas(df_filtrado, perfil)

    # Verificar que hay ejemplos de ambas clases
    if y.sum() == 0 or y.sum() == len(y):
        print("  ⚠️  Todas las etiquetas son iguales, ajustando umbrales...")
        y = (df_filtrado['calories'] <= df_filtrado['calories'].median()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = DecisionTreeClassifier(
        max_depth=5,          # Evita overfitting
        min_samples_leaf=10,  # Mínimo de muestras por hoja
        random_state=42
    )
    modelo.fit(X_train, y_train)

    y_pred   = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    reporte  = classification_report(y_test, y_pred,
                                     target_names=["No apto", "Apto"],
                                     zero_division=0)

    return modelo, accuracy, reporte


def aplicar_arbol(df_filtrado: pd.DataFrame, modelo: DecisionTreeClassifier) -> pd.DataFrame:
    """
    Aplica el árbol entrenado para clasificar todos los alimentos
    y retorna solo los marcados como aptos (predicción = 1).

    Parámetros:
        df_filtrado : DataFrame filtrado por reglas
        modelo      : árbol de decisión entrenado

    Retorna:
        DataFrame con solo los alimentos aptos
    """
    features = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar']
    X = df_filtrado[features].fillna(0)
    predicciones = modelo.predict(X)
    df_filtrado = df_filtrado.copy()
    df_filtrado['apto'] = predicciones
    return df_filtrado[df_filtrado['apto'] == 1].drop(columns=['apto']).reset_index(drop=True)


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def filtrar_alimentos(df: pd.DataFrame, perfil: dict,
                      restricciones: list = None,
                      max_cal: float = 800,
                      max_grasa: float = 50,
                      max_azucar: float = 40) -> tuple:
    """
    Función principal del módulo. Ejecuta el pipeline completo:
    1. Filtrado por reglas (restricciones + límites)
    2. Entrenamiento del árbol de decisión
    3. Clasificación final de alimentos aptos

    Parámetros:
        df           : DataFrame completo de foods_clean.csv
        perfil       : resultado de modules/profile.py → calcular_perfil()
        restricciones: lista de restricciones (ej: ['gluten', 'lactosa'])
        max_cal      : calorías máximas por 100g
        max_grasa    : grasa máxima por 100g
        max_azucar   : azúcar máxima por 100g

    Retorna:
        (df_aptos, modelo, accuracy)
    """
    if restricciones is None:
        restricciones = []

    print("\n📋 PASO 1: Filtrado por reglas...")
    df_reglas = filtrar_por_reglas(df, restricciones, max_cal, max_grasa, max_azucar)
    print(f"  ✅ Alimentos después de reglas: {len(df_reglas):,}")

    print("\n🌳 PASO 2: Entrenando árbol de decisión...")
    modelo, accuracy, reporte = entrenar_arbol(df_reglas, perfil)
    print(f"  ✅ Accuracy del árbol: {accuracy:.2%}")
    print(f"\n{reporte}")

    print("\n🔍 PASO 3: Clasificando alimentos aptos...")
    df_aptos = aplicar_arbol(df_reglas, modelo)
    print(f"  ✅ Alimentos aptos finales: {len(df_aptos):,}")

    return df_aptos, modelo, accuracy


# ─────────────────────────────────────────────
# PRUEBA RÁPIDA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.profile import calcular_perfil

    print("=" * 55)
    print("   PRUEBA DEL MÓDULO: filter.py")
    print("=" * 55)

    # Cargar dataset
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'processed', 'foods_clean.csv')
    df = pd.read_csv(DATA_PATH)
    print(f"\n📦 Dataset cargado: {len(df):,} alimentos")

    # Perfil de prueba
    perfil = calcular_perfil(
        peso_kg=70, altura_cm=175, edad=25,
        sexo="masculino", nivel_actividad="moderado",
        meta="perder peso"
    )
    print(f"\n👤 Perfil: {perfil['calorias_objetivo']} kcal/día | "
          f"P:{perfil['protein_g']}g C:{perfil['carbs_g']}g G:{perfil['fat_g']}g")

    # Filtrar con restricciones de ejemplo
    restricciones = ["gluten", "lactosa"]
    df_aptos, modelo, accuracy = filtrar_alimentos(
        df, perfil,
        restricciones=restricciones,
        max_cal=700,
        max_grasa=40,
        max_azucar=30
    )

    print("\n🍽️  Muestra de alimentos aptos:")
    print(df_aptos[['nombre', 'calories', 'protein', 'carbs', 'fat']].head(10).to_string(index=False))
    print(f"\n{'=' * 55}")
    print(f"  Total aptos: {len(df_aptos):,} alimentos")
    print(f"  Accuracy:    {accuracy:.2%}")
    print(f"{'=' * 55}")
