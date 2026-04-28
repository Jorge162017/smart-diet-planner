"""
modules/evaluator.py
====================
Módulo 5: Evaluación de Calidad del Plan Nutricional
CC3045 – Inteligencia Artificial | Smart Diet Planner

Calcula métricas objetivas para medir qué tan bueno es
el plan generado respecto al perfil del usuario.

Métricas:
  1. Desviación calórica diaria promedio (%)
  2. Cobertura de macronutrientes por día (%)
  3. Índice de diversidad alimentaria
  4. Precisión del filtrado (restricciones respetadas)
  5. Score nutricional general (0-100)
"""

import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────────
# MÉTRICAS INDIVIDUALES
# ─────────────────────────────────────────────

def desviacion_calorica(plan_df: pd.DataFrame, perfil: dict) -> dict:
    """
    Métrica 1: Desviación calórica diaria promedio.

    Mide qué tan cerca están las calorías de cada día
    respecto al objetivo del usuario.

    Objetivo: < 5% de desviación

    Retorna:
        Diccionario con desviación por día y promedio general
    """
    objetivo = perfil["calorias_objetivo"]
    resultado = {}

    calorias_por_dia = plan_df.groupby("dia")["calories"].sum()

    desviaciones = {}
    for dia, cal in calorias_por_dia.items():
        desv_pct = abs(cal - objetivo) / objetivo * 100
        desviaciones[dia] = {
            "calorias_reales":  round(cal, 1),
            "calorias_objetivo": objetivo,
            "desviacion_pct":   round(desv_pct, 2),
            "cumple":           desv_pct <= 10,   # tolerancia 10%
        }

    promedio_desv = np.mean([v["desviacion_pct"] for v in desviaciones.values()])

    resultado = {
        "por_dia":         desviaciones,
        "promedio_pct":    round(promedio_desv, 2),
        "dias_dentro_5pct": sum(1 for v in desviaciones.values()
                                if v["desviacion_pct"] <= 5),
        "dias_dentro_10pct": sum(1 for v in desviaciones.values()
                                 if v["desviacion_pct"] <= 10),
        "total_dias":      len(desviaciones),
        "estado":          "✅ Excelente" if promedio_desv <= 5
                           else "⚠️  Aceptable" if promedio_desv <= 10
                           else "❌ Mejorable",
    }
    return resultado


def cobertura_macros(plan_df: pd.DataFrame, perfil: dict) -> dict:
    """
    Métrica 2: Cobertura de macronutrientes.

    Calcula qué porcentaje de días cumple los objetivos
    de proteína, carbohidratos y grasa (con tolerancia ±15%).

    Retorna:
        Diccionario con cobertura por macro y promedio general
    """
    tolerancia = 0.15  # ±15%
    macros = {
        "protein": perfil["protein_g"],
        "carbs":   perfil["carbs_g"],
        "fat":     perfil["fat_g"],
    }

    totales_por_dia = plan_df.groupby("dia")[["protein", "carbs", "fat"]].sum()
    cobertura = {}

    for macro, objetivo in macros.items():
        limite_inf = objetivo * (1 - tolerancia)
        limite_sup = objetivo * (1 + tolerancia)
        dias_cumple = ((totales_por_dia[macro] >= limite_inf) &
                       (totales_por_dia[macro] <= limite_sup)).sum()
        pct = dias_cumple / len(totales_por_dia) * 100
        cobertura[macro] = {
            "objetivo_g":    round(objetivo, 1),
            "promedio_real": round(totales_por_dia[macro].mean(), 1),
            "dias_cumple":   int(dias_cumple),
            "cobertura_pct": round(pct, 1),
            "estado":        "✅" if pct >= 70 else "⚠️ " if pct >= 50 else "❌",
        }

    promedio_cobertura = np.mean([v["cobertura_pct"] for v in cobertura.values()])
    cobertura["promedio_general_pct"] = round(promedio_cobertura, 1)

    return cobertura


def diversidad_alimentaria(plan_df: pd.DataFrame) -> dict:
    """
    Métrica 3: Índice de diversidad alimentaria.

    Mide cuántos alimentos únicos aparecen en el plan
    respecto al total de apariciones.

    Un plan más diverso es más saludable y sostenible.

    Retorna:
        Diccionario con conteos y ratio de diversidad
    """
    total_apariciones = len(plan_df)
    alimentos_unicos  = plan_df["nombre"].nunique()
    ratio_diversidad  = alimentos_unicos / total_apariciones * 100

    # Diversidad por día
    diversidad_por_dia = plan_df.groupby("dia")["nombre"].nunique()

    return {
        "total_apariciones":   total_apariciones,
        "alimentos_unicos":    alimentos_unicos,
        "ratio_diversidad_pct": round(ratio_diversidad, 1),
        "promedio_unicos_dia": round(diversidad_por_dia.mean(), 1),
        "estado": "✅ Alta"    if ratio_diversidad >= 70
                  else "⚠️  Media" if ratio_diversidad >= 50
                  else "❌ Baja",
    }


def precision_filtrado(plan_df: pd.DataFrame, restricciones: list) -> dict:
    """
    Métrica 4: Precisión del filtrado de restricciones.

    Verifica que ningún alimento del plan viole las
    restricciones alimentarias del usuario.

    Objetivo: 100% de cumplimiento

    Retorna:
        Diccionario con violaciones encontradas y precisión
    """
    from modules.filter import RESTRICCIONES_KEYWORDS, contiene_restriccion

    if not restricciones:
        return {
            "restricciones":  [],
            "violaciones":    [],
            "total_alimentos": len(plan_df),
            "precision_pct":  100.0,
            "estado":         "✅ Sin restricciones",
        }

    violaciones = []
    for _, row in plan_df.iterrows():
        if contiene_restriccion(row["nombre"], restricciones):
            violaciones.append({
                "dia":    row["dia"],
                "comida": row["comida"],
                "nombre": row["nombre"],
            })

    total     = len(plan_df)
    precision = (total - len(violaciones)) / total * 100

    return {
        "restricciones":   restricciones,
        "violaciones":     violaciones,
        "total_alimentos": total,
        "precision_pct":   round(precision, 2),
        "estado":          "✅ Perfecto" if precision == 100
                           else f"❌ {len(violaciones)} violaciones",
    }


def score_general(desv: dict, cobertura: dict,
                  diversidad: dict, precision: dict) -> dict:
    """
    Métrica 5: Score nutricional general del plan (0-100).

    Pondera las 4 métricas anteriores en un único número
    que resume la calidad del plan.

    Pesos:
        - Desviación calórica: 30%
        - Cobertura de macros: 30%
        - Diversidad:          20%
        - Precisión filtrado:  20%

    Retorna:
        Diccionario con score final y desglose
    """
    # Convertir desviación a score (menor desviación = mayor score)
    score_cal  = max(0, 100 - desv["promedio_pct"] * 5)
    score_mac  = cobertura["promedio_general_pct"]
    score_div  = diversidad["ratio_diversidad_pct"]
    score_prec = precision["precision_pct"]

    score_final = (
        score_cal  * 0.30 +
        score_mac  * 0.30 +
        score_div  * 0.20 +
        score_prec * 0.20
    )

    return {
        "score_calorico":    round(score_cal, 1),
        "score_macros":      round(score_mac, 1),
        "score_diversidad":  round(score_div, 1),
        "score_precision":   round(score_prec, 1),
        "score_final":       round(score_final, 1),
        "calificacion":      "🏆 Excelente" if score_final >= 85
                             else "✅ Bueno"   if score_final >= 70
                             else "⚠️  Regular" if score_final >= 55
                             else "❌ Mejorar",
    }


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def evaluar_plan(plan_df: pd.DataFrame, perfil: dict,
                 restricciones: list = None) -> dict:
    """
    Función principal del módulo.
    Calcula todas las métricas de calidad del plan semanal.

    Parámetros:
        plan_df       : DataFrame del plan (output de planner.plan_a_dataframe)
        perfil        : diccionario del módulo profile.py
        restricciones : lista de restricciones del usuario

    Retorna:
        Diccionario completo con todas las métricas
    """
    if restricciones is None:
        restricciones = []

    print("\n📊 Evaluando calidad del plan...")

    desv      = desviacion_calorica(plan_df, perfil)
    cobertura = cobertura_macros(plan_df, perfil)
    diversidad = diversidad_alimentaria(plan_df)
    precision  = precision_filtrado(plan_df, restricciones)
    score      = score_general(desv, cobertura, diversidad, precision)

    return {
        "desviacion_calorica": desv,
        "cobertura_macros":    cobertura,
        "diversidad":          diversidad,
        "precision_filtrado":  precision,
        "score":               score,
    }


def imprimir_reporte(metricas: dict):
    """
    Imprime un reporte legible de las métricas en consola.

    Parámetros:
        metricas : diccionario output de evaluar_plan()
    """
    print("\n" + "=" * 55)
    print("   📋 REPORTE DE CALIDAD DEL PLAN")
    print("=" * 55)

    # Score general
    s = metricas["score"]
    print(f"\n🏅 SCORE GENERAL: {s['score_final']}/100  {s['calificacion']}")
    print(f"   Calórico:   {s['score_calorico']}/100")
    print(f"   Macros:     {s['score_macros']}/100")
    print(f"   Diversidad: {s['score_diversidad']}/100")
    print(f"   Precisión:  {s['score_precision']}/100")

    # Desviación calórica
    d = metricas["desviacion_calorica"]
    print(f"\n🔥 DESVIACIÓN CALÓRICA: {d['promedio_pct']}%  {d['estado']}")
    print(f"   Días dentro del 5%:  {d['dias_dentro_5pct']}/{d['total_dias']}")
    print(f"   Días dentro del 10%: {d['dias_dentro_10pct']}/{d['total_dias']}")

    # Cobertura macros
    c = metricas["cobertura_macros"]
    print(f"\n💪 COBERTURA DE MACROS: {c['promedio_general_pct']}%")
    for macro in ["protein", "carbs", "fat"]:
        m = c[macro]
        print(f"   {macro:8}: objetivo {m['objetivo_g']}g | "
              f"real {m['promedio_real']}g | "
              f"{m['cobertura_pct']}% días OK  {m['estado']}")

    # Diversidad
    div = metricas["diversidad"]
    print(f"\n🌈 DIVERSIDAD: {div['alimentos_unicos']} alimentos únicos  {div['estado']}")
    print(f"   Ratio: {div['ratio_diversidad_pct']}% | "
          f"Promedio por día: {div['promedio_unicos_dia']} alimentos")

    # Precisión filtrado
    p = metricas["precision_filtrado"]
    print(f"\n🛡️  PRECISIÓN FILTRADO: {p['precision_pct']}%  {p['estado']}")
    if p["violaciones"]:
        print("   Violaciones encontradas:")
        for v in p["violaciones"]:
            print(f"   ⚠️  {v['dia']} {v['comida']}: {v['nombre']}")

    print("\n" + "=" * 55)


# ─────────────────────────────────────────────
# PRUEBA RÁPIDA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.profile     import calcular_perfil
    from modules.filter      import filtrar_alimentos
    from modules.recommender import get_recomendaciones
    from modules.planner     import generar_plan_semanal, plan_a_dataframe

    print("=" * 55)
    print("   PRUEBA DEL MÓDULO: evaluator.py")
    print("=" * 55)

    # Pipeline completo
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "processed", "foods_clean.csv")
    df = pd.read_csv(DATA_PATH)

    restricciones = ["gluten", "lactosa"]
    perfil = calcular_perfil(70, 175, 25, "masculino", "moderado", "perder peso")

    df_aptos, _, _ = filtrar_alimentos(
        df, perfil, restricciones=restricciones,
        max_cal=700, max_grasa=40, max_azucar=30
    )
    df_rec, _, _ = get_recomendaciones(
        df_aptos, perfil,
        preferencias=["chicken", "salmon"],
        k=8, n_total=80
    )

    plan    = generar_plan_semanal(df_rec, perfil, dias=7)
    plan_df = plan_a_dataframe(plan)

    # Evaluar
    metricas = evaluar_plan(plan_df, perfil, restricciones=restricciones)
    imprimir_reporte(metricas)