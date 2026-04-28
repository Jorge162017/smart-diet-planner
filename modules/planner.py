"""
modules/planner.py
==================
Generación del Plan Alimenticio Semanal

Técnica: Agente Inteligente con Búsqueda Greedy
El agente percibe el perfil nutricional del usuario y selecciona
iterativamente los mejores alimentos para cada tiempo de comida,
minimizando la desviación respecto a los objetivos diarios.

Plan generado: 7 días × 4 comidas (desayuno, almuerzo, cena, snack)
"""

import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

COMIDAS = ["Desayuno", "Almuerzo", "Cena", "Snack"]

DISTRIBUCION_CALORIAS = {
    "Desayuno": 0.25,
    "Almuerzo": 0.35,
    "Cena":     0.30,
    "Snack":    0.10,
}


# ─────────────────────────────────────────────
# PORCIÓN ADAPTATIVA
# ─────────────────────────────────────────────

def calcular_gramos_optimos(alimento: pd.Series, cal_objetivo: float) -> float:
    """
    Calcula cuántos gramos se necesitan para alcanzar
    las calorías objetivo de la comida.
    Limita entre 80g y 400g para que sea una porción realista.
    """
    cal_por_100g = alimento["calories"]
    if cal_por_100g <= 0:
        return 150.0
    gramos = (cal_objetivo / cal_por_100g) * 100
    return round(max(80, min(gramos, 400)), 1)


def calcular_nutrientes_porcion(alimento: pd.Series, gramos: float) -> dict:
    """
    Convierte valores nutricionales por 100g a la porción real.
    """
    f = gramos / 100
    return {
        "calories": round(alimento["calories"] * f, 1),
        "protein":  round(alimento["protein"]  * f, 1),
        "carbs":    round(alimento["carbs"]    * f, 1),
        "fat":      round(alimento["fat"]      * f, 1),
        "fiber":    round(alimento["fiber"]    * f, 1),
    }


# ─────────────────────────────────────────────
# SCORE GREEDY
# ─────────────────────────────────────────────

def score_greedy(alimento: pd.Series, cal_objetivo: float,
                 prot_objetivo: float) -> float:
    """
    Puntuación greedy para seleccionar el mejor alimento.

    Premia acercarse a las calorías objetivo y tener fibra.
    Penaliza el exceso de proteína sobre el objetivo de la comida.
    """
    gramos = calcular_gramos_optimos(alimento, cal_objetivo)
    n      = calcular_nutrientes_porcion(alimento, gramos)

    exceso_prot = max(0, n["protein"] - prot_objetivo * 1.3)

    score = (
        - abs(n["calories"] - cal_objetivo) * 0.5
        + n["fiber"] * 2.0
        - exceso_prot * 2.0
        + min(n["protein"], prot_objetivo) * 0.5
    )
    return score


# ─────────────────────────────────────────────
# SELECCIÓN DE COMIDA
# ─────────────────────────────────────────────

def seleccionar_comida(df_candidatos: pd.DataFrame,
                       cal_objetivo: float, prot_objetivo: float,
                       usados_hoy: set, top_n: int = 6) -> tuple:
    """
    Evalúa candidatos con el score greedy, toma el top_n
    y elige aleatoriamente entre ellos para dar variedad.

    Retorna:
        (alimento seleccionado, gramos óptimos)
    """
    candidatos = df_candidatos[~df_candidatos["nombre"].isin(usados_hoy)].copy()
    if len(candidatos) < 3:
        candidatos = df_candidatos.copy()

    candidatos["_score"] = candidatos.apply(
        lambda row: score_greedy(row, cal_objetivo, prot_objetivo), axis=1
    )

    top     = candidatos.nlargest(min(top_n, len(candidatos)), "_score")
    elegido = top.sample(1, random_state=np.random.randint(0, 9999)).iloc[0]
    gramos  = calcular_gramos_optimos(elegido, cal_objetivo)

    return elegido, gramos


# ─────────────────────────────────────────────
# GENERACIÓN POR DÍA
# ─────────────────────────────────────────────

def generar_dia(df_recomendados: pd.DataFrame, perfil: dict,
                usados_semana: set) -> dict:
    """
    Genera el plan de un día completo (4 comidas)
    usando el agente greedy con porciones adaptativas.
    """
    plan_dia = {}
    totales  = {"calories": 0.0, "protein": 0.0,
                "carbs": 0.0, "fat": 0.0, "fiber": 0.0}
    usados_hoy = set()

    # Preferir alimentos no usados esta semana
    frescos = df_recomendados[~df_recomendados["nombre"].isin(usados_semana)]
    pool    = frescos if len(frescos) >= 10 else df_recomendados

    for comida in COMIDAS:
        cal_objetivo  = perfil["calorias_objetivo"] * DISTRIBUCION_CALORIAS[comida]
        prot_objetivo = perfil["protein_g"]         * DISTRIBUCION_CALORIAS[comida]

        alimento, gramos = seleccionar_comida(
            pool, cal_objetivo, prot_objetivo, usados_hoy, top_n=6
        )
        nutrientes = calcular_nutrientes_porcion(alimento, gramos)

        plan_dia[comida] = {"nombre": alimento["nombre"], "gramos": gramos, **nutrientes}

        for key in totales:
            totales[key] = round(totales[key] + nutrientes[key], 1)

        usados_hoy.add(alimento["nombre"])
        usados_semana.add(alimento["nombre"])

    plan_dia["_totales"] = totales
    return plan_dia


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def generar_plan_semanal(df_recomendados: pd.DataFrame,
                          perfil: dict, dias: int = 7) -> dict:
    """
    Genera el plan de alimentación completo para N días.
    """
    print(f"\n🗓️  Generando plan de {dias} días...")
    print(f"   Objetivo: {perfil['calorias_objetivo']} kcal | "
          f"P:{perfil['protein_g']}g C:{perfil['carbs_g']}g G:{perfil['fat_g']}g")

    plan          = {}
    usados_semana = set()
    nombres_dias  = ["Lunes", "Martes", "Miércoles", "Jueves",
                     "Viernes", "Sábado", "Domingo"]

    for i in range(dias):
        dia              = nombres_dias[i]
        plan[dia]        = generar_dia(df_recomendados, perfil, usados_semana)
        t                = plan[dia]["_totales"]
        desv_pct         = abs(t["calories"] - perfil["calorias_objetivo"]) \
                           / perfil["calorias_objetivo"] * 100
        print(f"  ✅ {dia}: {t['calories']} kcal (desv {desv_pct:.1f}%) | "
              f"P:{t['protein']}g C:{t['carbs']}g G:{t['fat']}g")

    print(f"\n  ✅ Plan generado con {len(usados_semana)} alimentos únicos")
    return plan


def plan_a_dataframe(plan: dict) -> pd.DataFrame:
    """Convierte el plan semanal a DataFrame tabular."""
    filas = []
    for dia, comidas in plan.items():
        for comida, datos in comidas.items():
            if comida == "_totales":
                continue
            filas.append({"dia": dia, "comida": comida,
                          "nombre": datos["nombre"], "gramos": datos["gramos"],
                          "calories": datos["calories"], "protein": datos["protein"],
                          "carbs": datos["carbs"], "fat": datos["fat"],
                          "fiber": datos["fiber"]})
    return pd.DataFrame(filas)


def lista_compras(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Genera la lista de compras consolidada del plan semanal."""
    return (
        plan_df.groupby("nombre")
        .agg(veces=("nombre", "count"),
             gramos_total=("gramos", "sum"),
             cal_total=("calories", "sum"))
        .sort_values("gramos_total", ascending=False)
        .reset_index()
        .rename(columns={"nombre": "alimento"})
    )


# ─────────────────────────────────────────────
# PRUEBA RÁPIDA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.profile     import calcular_perfil
    from modules.filter      import filtrar_alimentos
    from modules.recommender import get_recomendaciones

    print("=" * 60)
    print("   PRUEBA DEL MÓDULO: planner.py (porciones adaptativas)")
    print("=" * 60)

    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "processed", "foods_clean.csv")
    df     = pd.read_csv(DATA_PATH)
    perfil = calcular_perfil(70, 175, 25, "masculino", "moderado", "perder peso")

    df_aptos, _, _ = filtrar_alimentos(
        df, perfil, restricciones=["gluten", "lactosa"],
        max_cal=700, max_grasa=40, max_azucar=30
    )
    df_rec, _, _ = get_recomendaciones(
        df_aptos, perfil, preferencias=["chicken", "salmon"], k=8, n_total=80
    )

    plan    = generar_plan_semanal(df_rec, perfil, dias=7)
    plan_df = plan_a_dataframe(plan)

    print("\n📋 PLAN — primeros 3 días:")
    print("=" * 60)
    for dia in ["Lunes", "Martes", "Miércoles"]:
        print(f"\n📅 {dia}:")
        for _, row in plan_df[plan_df["dia"] == dia].iterrows():
            print(f"  {row['comida']:10} | {row['nombre'][:38]:38} | "
                  f"{row['gramos']:5.0f}g | {row['calories']:5.0f} kcal | "
                  f"P:{row['protein']:4.1f}g")
        t = plan[dia]["_totales"]
        desv_pct = abs(t["calories"] - perfil["calorias_objetivo"]) \
                   / perfil["calorias_objetivo"] * 100
        print(f"  {'TOTAL':10}   {'':38}         "
              f"{t['calories']:5.0f} kcal  (desv {desv_pct:.1f}%)")

    compras = lista_compras(plan_df)
    PROCESSED = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "processed")
    plan_df.to_csv(os.path.join(PROCESSED, "plan_semanal.csv"), index=False)
    compras.to_csv(os.path.join(PROCESSED, "lista_compras.csv"), index=False)
    print(f"\n✅ CSVs guardados | {len(compras)} alimentos únicos en la lista")
    print("=" * 60)