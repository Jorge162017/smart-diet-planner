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

# Palabras clave para descartar alimentos que no son comidas principales:
# especias secas, hierbas, levaduras, proteínas concentradas artificiales,
# carnes silvestres desecadas y bebidas.
KEYWORDS_NO_COMIDA = [
    "spices,", "spice,",
    "leavening agents", "leavening",
    "freeze-dried", "freeze dried",
    "seasonings,", "seasoning,",
    "spearmint", "peppermint",
    "dill weed", "dill seed", "fenugreek seed", "fennel seed",
    "caraway seed", "anise seed", "cumin seed", "coriander seed",
    "chives,", "parsley, dried",
    "tarragon, dried", "rosemary,", "sage,", "thyme,",
    "gelatins, dry", "gelatin, dry", "gelatin desserts, dry",
    "soy protein isolate", "soy protein concentrate",
    "soy meal, defatted",
    "whale,", "beluga,", "walrus,", "seal,",
    "caribou,",
    "baker's yeast", "active dry yeast",
    "meat extender",
    "beverages, tea",
    "safflower seed meal",
]


# ─────────────────────────────────────────────
# FILTRO DE ALIMENTOS REALISTAS
# ─────────────────────────────────────────────

def es_comida_realista(nombre: str) -> bool:
    """
    Retorna False si el alimento es una especia, hierba seca, levadura,
    proteína concentrada artificial, carne silvestre desecada u otro
    ingrediente no apto como comida principal de un plan nutricional.
    """
    n = nombre.lower()
    return not any(kw in n for kw in KEYWORDS_NO_COMIDA)


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
                 prot_objetivo: float, carbs_objetivo: float,
                 fat_objetivo: float) -> float:
    """
    Puntuación greedy que balancea calorías y los tres macronutrientes.

    Cada desviación se normaliza por el objetivo (fracción 0–1+) para
    que los cuatro términos sean comparables en escala.
    Pesos: calorías 35%, proteína 25%, carbohidratos 20%, grasa 20%.
    Bonus moderado de fibra.
    """
    gramos = calcular_gramos_optimos(alimento, cal_objetivo)
    n      = calcular_nutrientes_porcion(alimento, gramos)

    def desv_norm(real: float, obj: float) -> float:
        return abs(real - obj) / max(obj, 1.0)

    score = (
        - desv_norm(n["calories"], cal_objetivo)   * 35.0
        - desv_norm(n["protein"],  prot_objetivo)  * 25.0
        - desv_norm(n["carbs"],    carbs_objetivo) * 20.0
        - desv_norm(n["fat"],      fat_objetivo)   * 20.0
        + n["fiber"] * 0.3
    )
    return score


# ─────────────────────────────────────────────
# SELECCIÓN DE COMIDA
# ─────────────────────────────────────────────

def seleccionar_comida(df_candidatos: pd.DataFrame,
                       cal_objetivo: float, prot_objetivo: float,
                       carbs_objetivo: float, fat_objetivo: float,
                       usados_hoy: set, top_n: int = 6) -> tuple:
    """
    Excluye alimentos no realistas, evalúa con el score greedy de cuatro
    macros, toma el top_n y elige aleatoriamente entre ellos para variedad.

    Retorna:
        (alimento seleccionado, gramos óptimos)
    """
    candidatos = df_candidatos[~df_candidatos["nombre"].isin(usados_hoy)].copy()
    if len(candidatos) < 3:
        candidatos = df_candidatos.copy()

    # Excluir ingredientes/especias/concentrados no aptos como comida
    candidatos_reales = candidatos[
        candidatos["nombre"].apply(es_comida_realista)
    ].copy()
    if len(candidatos_reales) >= 5:
        candidatos = candidatos_reales

    candidatos["_score"] = candidatos.apply(
        lambda row: score_greedy(
            row, cal_objetivo, prot_objetivo, carbs_objetivo, fat_objetivo
        ),
        axis=1,
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
    Genera el plan de un día completo (4 comidas).

    Las calorías se distribuyen con porcentajes fijos (25/35/30/10 %).
    Los objetivos de macros son adaptativos: tras cada comida se
    redistribuyen los macros restantes entre las comidas pendientes,
    compensando excesos o déficits acumulados durante el día.
    Pisos mínimos evitan penalizaciones extremas al final del día.
    """
    plan_dia = {}
    totales  = {"calories": 0.0, "protein": 0.0,
                "carbs": 0.0, "fat": 0.0, "fiber": 0.0}
    usados_hoy = set()

    # Macros totales aún pendientes para el día
    macro_rest = {
        "protein": perfil["protein_g"],
        "carbs":   perfil["carbs_g"],
        "fat":     perfil["fat_g"],
    }

    # Preferir alimentos no usados esta semana
    frescos = df_recomendados[~df_recomendados["nombre"].isin(usados_semana)]
    pool    = frescos if len(frescos) >= 10 else df_recomendados

    n_rest = len(COMIDAS)

    for comida in COMIDAS:
        # Calorías: distribución fija para mantener tamaño de comida realista
        cal_objetivo = perfil["calorias_objetivo"] * DISTRIBUCION_CALORIAS[comida]

        # Macros: redistribuir lo que falta entre las comidas que quedan.
        # max() con un piso mínimo evita objetivos negativos o extremos al
        # compensar excesos muy grandes al final del día.
        prot_objetivo  = max(macro_rest["protein"] / n_rest, 5.0)
        carbs_objetivo = max(macro_rest["carbs"]   / n_rest, 5.0)
        fat_objetivo   = max(macro_rest["fat"]     / n_rest, 2.0)

        alimento, gramos = seleccionar_comida(
            pool, cal_objetivo, prot_objetivo,
            carbs_objetivo, fat_objetivo,
            usados_hoy, top_n=6
        )
        nutrientes = calcular_nutrientes_porcion(alimento, gramos)

        plan_dia[comida] = {"nombre": alimento["nombre"], "gramos": gramos, **nutrientes}

        for key in totales:
            totales[key] = round(totales[key] + nutrientes[key], 1)

        # Descontar lo consumido de los macros pendientes
        macro_rest["protein"] -= nutrientes["protein"]
        macro_rest["carbs"]   -= nutrientes["carbs"]
        macro_rest["fat"]     -= nutrientes["fat"]
        n_rest -= 1

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
    sys.stdout.reconfigure(encoding='utf-8')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules.profile     import calcular_perfil
    from modules.filter      import filtrar_alimentos
    from modules.recommender import get_recomendaciones

    print("=" * 75)
    print("   PRUEBA DEL MÓDULO: planner.py (agente greedy mejorado)")
    print("=" * 75)

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
    print("=" * 75)
    for dia in ["Lunes", "Martes", "Miércoles"]:
        print(f"\n📅 {dia}:")
        for _, row in plan_df[plan_df["dia"] == dia].iterrows():
            print(f"  {row['comida']:10} | {row['nombre'][:35]:35} | "
                  f"{row['gramos']:5.0f}g | {row['calories']:5.0f} kcal | "
                  f"P:{row['protein']:5.1f}g C:{row['carbs']:5.1f}g G:{row['fat']:5.1f}g")
        t = plan[dia]["_totales"]
        desv_pct = abs(t["calories"] - perfil["calorias_objetivo"]) \
                   / perfil["calorias_objetivo"] * 100
        print(f"  {'TOTAL':10}   {'':35}          "
              f"{t['calories']:5.0f} kcal (desv {desv_pct:.1f}%) | "
              f"P:{t['protein']:5.1f}g C:{t['carbs']:5.1f}g G:{t['fat']:5.1f}g")

    compras = lista_compras(plan_df)
    PROCESSED = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "processed")
    plan_df.to_csv(os.path.join(PROCESSED, "plan_semanal.csv"), index=False)
    compras.to_csv(os.path.join(PROCESSED, "lista_compras.csv"), index=False)
    print(f"\n✅ CSVs guardados | {len(compras)} alimentos únicos en la lista")
    print("=" * 75)