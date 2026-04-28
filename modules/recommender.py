"""
modules/recommender.py
======================
Recomendación Personalizada de Alimentos

Técnica: K-Nearest Neighbors (KNN)
Encuentra alimentos similares a los preferidos por el usuario
dentro del conjunto de alimentos aptos, usando distancia
euclidiana en el espacio de vectores nutricionales.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os


# ─────────────────────────────────────────────
# FEATURES NUTRICIONALES PARA KNN
# ─────────────────────────────────────────────
FEATURES = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar']


# ─────────────────────────────────────────────
# FUNCIONES PRINCIPALES
# ─────────────────────────────────────────────

def entrenar_knn(df_aptos: pd.DataFrame, k: int = 5) -> tuple:
    """
    Entrena el modelo KNN sobre el espacio nutricional
    de los alimentos aptos.

    Los valores se normalizan con StandardScaler para que
    ningún nutriente domine por su escala (ej: calorías vs fibra).

    Parámetros:
        df_aptos : DataFrame de alimentos aptos (output de filter.py)
        k        : número de vecinos a buscar

    Retorna:
        (modelo KNN entrenado, scaler ajustado)
    """
    X = df_aptos[FEATURES].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo_knn = NearestNeighbors(
        n_neighbors=k + 1,   # +1 porque el propio alimento aparece como vecino
        metric='euclidean',
        algorithm='auto'
    )
    modelo_knn.fit(X_scaled)

    return modelo_knn, scaler


def buscar_similares(nombre_alimento: str, df_aptos: pd.DataFrame,
                     modelo_knn: NearestNeighbors,
                     scaler: StandardScaler, k: int = 5) -> pd.DataFrame:
    """
    Dado el nombre de un alimento preferido, encuentra los k
    alimentos más similares nutricionalmente en el dataset apto.

    Parámetros:
        nombre_alimento : nombre parcial o completo del alimento
        df_aptos        : DataFrame de alimentos aptos
        modelo_knn      : modelo KNN entrenado
        scaler          : StandardScaler ajustado
        k               : número de similares a retornar

    Retorna:
        DataFrame con los k alimentos más similares
    """
    # Buscar el alimento en el dataset (búsqueda parcial, case-insensitive)
    matches = df_aptos[df_aptos['nombre'].str.lower().str.contains(
        nombre_alimento.lower(), na=False
    )]

    if matches.empty:
        print(f"  ⚠️  '{nombre_alimento}' no encontrado en el dataset apto.")
        return pd.DataFrame()

    # Tomar el primer match
    alimento_ref = matches.iloc[0]
    vector = alimento_ref[FEATURES].fillna(0).values.reshape(1, -1)
    vector_scaled = scaler.transform(vector)

    # Buscar los k vecinos más cercanos
    distancias, indices = modelo_knn.kneighbors(vector_scaled)

    # Excluir el propio alimento (índice 0)
    indices_similares = indices[0][1:k+1]
    distancias_similares = distancias[0][1:k+1]

    resultado = df_aptos.iloc[indices_similares].copy()
    resultado['distancia'] = distancias_similares.round(3)
    resultado['similitud_pct'] = (
        (1 / (1 + distancias_similares)) * 100
    ).round(1)

    return resultado[['nombre', 'calories', 'protein', 'carbs',
                       'fat', 'fiber', 'distancia', 'similitud_pct']]


def recomendar_por_preferencias(preferencias: list, df_aptos: pd.DataFrame,
                                 modelo_knn: NearestNeighbors,
                                 scaler: StandardScaler,
                                 k_por_preferencia: int = 10) -> pd.DataFrame:
    """
    Genera recomendaciones a partir de una lista de alimentos
    preferidos por el usuario. Para cada preferencia busca k
    similares y combina todos los resultados sin duplicados.

    Parámetros:
        preferencias       : lista de nombres de alimentos preferidos
        df_aptos           : DataFrame de alimentos aptos
        modelo_knn         : modelo KNN entrenado
        scaler             : StandardScaler ajustado
        k_por_preferencia  : similares a buscar por cada preferencia

    Retorna:
        DataFrame con todas las recomendaciones únicas, ordenadas
        por similitud promedio descendente
    """
    todos = []

    for pref in preferencias:
        similares = buscar_similares(pref, df_aptos, modelo_knn, scaler, k_por_preferencia)
        if not similares.empty:
            similares['preferencia_origen'] = pref
            todos.append(similares)
            print(f"  ✅ '{pref}' → {len(similares)} similares encontrados")
        else:
            print(f"  ⚠️  '{pref}' → sin coincidencias")

    if not todos:
        print("  ⚠️  Ninguna preferencia encontrada, retornando alimentos aptos aleatorios.")
        return df_aptos.sample(min(50, len(df_aptos))).reset_index(drop=True)

    df_recomendados = pd.concat(todos, ignore_index=True)

    # Eliminar duplicados, quedarse con el de mayor similitud
    df_recomendados = (
        df_recomendados
        .sort_values('similitud_pct', ascending=False)
        .drop_duplicates(subset='nombre')
        .reset_index(drop=True)
    )

    return df_recomendados


def recomendar_sin_preferencias(df_aptos: pd.DataFrame, perfil: dict,
                                 n: int = 100) -> pd.DataFrame:
    """
    Cuando el usuario no tiene preferencias definidas, selecciona
    los mejores alimentos basándose en un score nutricional
    alineado con el perfil del usuario.

    Score = proteína * peso_meta - |calorías - cal_objetivo_comida|

    Parámetros:
        df_aptos : DataFrame de alimentos aptos
        perfil   : diccionario del módulo profile.py
        n        : número de alimentos a retornar

    Retorna:
        DataFrame con los n mejores alimentos según el perfil
    """
    cal_por_comida  = perfil['calorias_objetivo'] / 4
    prot_por_comida = perfil['protein_g'] / 4

    # Factor de proteína según meta
    peso_proteina = {
        "perder peso":   2.0,
        "ganar músculo": 2.5,
        "mantenerse":    1.5,
    }.get(perfil['meta'], 1.5)

    df_scored = df_aptos.copy()
    df_scored['score'] = (
        df_scored['protein'] * peso_proteina
        - abs(df_scored['calories'] - (cal_por_comida / 150 * 100))
        + df_scored['fiber'] * 1.5
    )

    return (
        df_scored
        .nlargest(n, 'score')
        .reset_index(drop=True)
    )


def get_recomendaciones(df_aptos: pd.DataFrame, perfil: dict,
                         preferencias: list = None, k: int = 5,
                         n_total: int = 80) -> tuple:
    """
    Función principal del módulo.
    Retorna un pool de alimentos recomendados para el planificador.

    Parámetros:
        df_aptos     : alimentos aptos (output de filter.py)
        perfil       : diccionario del módulo profile.py
        preferencias : lista de alimentos preferidos (puede ser vacía)
        k            : vecinos por preferencia en KNN
        n_total      : tamaño máximo del pool de recomendados

    Retorna:
        (df_recomendados, modelo_knn, scaler)
    """
    print(f"\n🤖 Entrenando KNN con {len(df_aptos):,} alimentos aptos...")
    modelo_knn, scaler = entrenar_knn(df_aptos, k=k)
    print(f"  ✅ KNN entrenado (k={k}, métrica=euclidiana)")

    if preferencias:
        print(f"\n🔎 Buscando similares a preferencias: {preferencias}")
        df_recomendados = recomendar_por_preferencias(
            preferencias, df_aptos, modelo_knn, scaler, k_por_preferencia=k
        )
        # Completar con los mejores según perfil si hay pocos
        if len(df_recomendados) < n_total:
            extras = recomendar_sin_preferencias(df_aptos, perfil, n=n_total)
            df_recomendados = pd.concat([df_recomendados, extras], ignore_index=True)
            df_recomendados = df_recomendados.drop_duplicates(subset='nombre').head(n_total)
    else:
        print("\n📋 Sin preferencias → seleccionando por perfil nutricional...")
        df_recomendados = recomendar_sin_preferencias(df_aptos, perfil, n=n_total)

    df_recomendados = df_recomendados.reset_index(drop=True)
    print(f"\n  ✅ Pool de recomendados: {len(df_recomendados):,} alimentos listos para el plan")

    return df_recomendados, modelo_knn, scaler


# ─────────────────────────────────────────────
# PRUEBA RÁPIDA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.profile import calcular_perfil
    from modules.filter import filtrar_alimentos

    print("=" * 55)
    print("   PRUEBA DEL MÓDULO: recommender.py")
    print("=" * 55)

    # Cargar dataset
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'processed', 'foods_clean.csv')
    df = pd.read_csv(DATA_PATH)
    print(f"\n📦 Dataset: {len(df):,} alimentos")

    # Perfil
    perfil = calcular_perfil(70, 175, 25, "masculino", "moderado", "perder peso")
    print(f"👤 Perfil: {perfil['calorias_objetivo']} kcal/día")

    # Filtrar
    df_aptos, modelo_dt, acc = filtrar_alimentos(
        df, perfil, restricciones=["gluten", "lactosa"],
        max_cal=700, max_grasa=40, max_azucar=30
    )

    # Recomendar CON preferencias
    preferencias = ["chicken", "salmon", "rice"]
    df_rec, knn, scaler = get_recomendaciones(
        df_aptos, perfil, preferencias=preferencias, k=8, n_total=60
    )

    print("\n🍽️  Top 10 alimentos recomendados:")
    cols_show = ['nombre', 'calories', 'protein', 'carbs', 'fat']
    print(df_rec[cols_show].head(10).to_string(index=False))

    # Probar similares a un alimento específico
    print("\n\n🔍 Alimentos similares a 'chicken':")
    modelo_solo, scaler_solo = entrenar_knn(df_aptos, k=5)
    similares = buscar_similares("chicken", df_aptos, modelo_solo, scaler_solo, k=5)
    if not similares.empty:
        print(similares.to_string(index=False))

    print(f"\n{'=' * 55}")
    print(f"  Recomendados totales: {len(df_rec):,}")
    print(f"{'=' * 55}")
