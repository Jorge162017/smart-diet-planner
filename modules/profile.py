"""
modules/profile.py
==================
Cálculo del Perfil Nutricional del Usuario

Técnica: Regresión 
Calcula el TDEE (Total Daily Energy Expenditure) del usuario y sus
macronutrientes objetivo según su meta de salud.
"""


# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

# Factor de actividad física
ACTIVITY_FACTORS = {
    "sedentario":   1.2,    # Poco o ningún ejercicio
    "ligero":       1.375,  # Ejercicio 1-3 días/semana
    "moderado":     1.55,   # Ejercicio 3-5 días/semana
    "activo":       1.725,  # Ejercicio 6-7 días/semana
    "muy activo":   1.9,    # Ejercicio muy intenso o trabajo físico
}

# Distribución de macros según meta (% de calorías totales)
MACRO_GOALS = {
    "perder peso": {
        "protein_pct": 0.35,   # Alta proteína para preservar músculo
        "carbs_pct":   0.35,
        "fat_pct":     0.30,
        "calorie_adj": -500,   # Déficit de 500 kcal
    },
    "ganar músculo": {
        "protein_pct": 0.30,
        "carbs_pct":   0.45,
        "fat_pct":     0.25,
        "calorie_adj": +300,   # Superávit de 300 kcal
    },
    "mantenerse": {
        "protein_pct": 0.25,
        "carbs_pct":   0.50,
        "fat_pct":     0.25,
        "calorie_adj": 0,
    },
}


# ─────────────────────────────────────────────
# FUNCIONES PRINCIPALES
# ─────────────────────────────────────────────

def calcular_bmr(peso_kg: float, altura_cm: float, edad: int, sexo: str) -> float:
    """
    Calcula la Tasa Metabólica Basal (BMR) usando la fórmula
    de Mifflin-St Jeor (versión revisada de Harris-Benedict).

    Parámetros:
        peso_kg   : peso en kilogramos
        altura_cm : altura en centímetros
        edad      : edad en años
        sexo      : 'masculino' o 'femenino'

    Retorna:
        BMR en kcal/día
    """
    if sexo.lower() == "masculino":
        bmr = (10 * peso_kg) + (6.25 * altura_cm) - (5 * edad) + 5
    else:
        bmr = (10 * peso_kg) + (6.25 * altura_cm) - (5 * edad) - 161

    return round(bmr, 2)


def calcular_tdee(bmr: float, nivel_actividad: str) -> float:
    """
    Calcula el Gasto Energético Total Diario (TDEE).

    Parámetros:
        bmr             : tasa metabólica basal (kcal/día)
        nivel_actividad : clave del diccionario ACTIVITY_FACTORS

    Retorna:
        TDEE en kcal/día
    """
    factor = ACTIVITY_FACTORS.get(nivel_actividad.lower(), 1.2)
    return round(bmr * factor, 2)


def calcular_calorias_objetivo(tdee: float, meta: str) -> float:
    """
    Ajusta el TDEE según la meta del usuario (déficit o superávit).

    Parámetros:
        tdee : gasto energético total diario
        meta : 'perder peso', 'ganar músculo' o 'mantenerse'

    Retorna:
        Calorías objetivo diarias
    """
    ajuste = MACRO_GOALS.get(meta.lower(), MACRO_GOALS["mantenerse"])["calorie_adj"]
    calorias = tdee + ajuste
    # No bajar de 1200 kcal (mínimo saludable)
    return round(max(calorias, 1200), 2)


def calcular_macros(calorias_objetivo: float, meta: str) -> dict:
    """
    Calcula los gramos de proteína, carbohidratos y grasa
    a partir de las calorías objetivo y la meta del usuario.

    Calorías por gramo: proteína=4, carbos=4, grasa=9

    Parámetros:
        calorias_objetivo : calorías diarias objetivo
        meta              : 'perder peso', 'ganar músculo' o 'mantenerse'

    Retorna:
        Diccionario con gramos de cada macro
    """
    distrib = MACRO_GOALS.get(meta.lower(), MACRO_GOALS["mantenerse"])

    protein_kcal = calorias_objetivo * distrib["protein_pct"]
    carbs_kcal   = calorias_objetivo * distrib["carbs_pct"]
    fat_kcal     = calorias_objetivo * distrib["fat_pct"]

    return {
        "protein_g": round(protein_kcal / 4, 1),
        "carbs_g":   round(carbs_kcal / 4, 1),
        "fat_g":     round(fat_kcal / 9, 1),
    }


def calcular_perfil(peso_kg: float, altura_cm: float, edad: int,
                    sexo: str, nivel_actividad: str, meta: str) -> dict:
    """
    Función principal del módulo.
    Calcula el perfil nutricional completo del usuario.

    Parámetros:
        peso_kg         : peso en kg
        altura_cm       : altura en cm
        edad            : edad en años
        sexo            : 'masculino' o 'femenino'
        nivel_actividad : 'sedentario', 'ligero', 'moderado', 'activo', 'muy activo'
        meta            : 'perder peso', 'ganar músculo', 'mantenerse'

    Retorna:
        Diccionario con BMR, TDEE, calorías objetivo y macros
    """
    # Validaciones básicas
    assert 10 <= peso_kg <= 300,    "Peso fuera de rango (10-300 kg)"
    assert 100 <= altura_cm <= 250, "Altura fuera de rango (100-250 cm)"
    assert 5 <= edad <= 120,        "Edad fuera de rango (5-120 años)"
    assert sexo.lower() in ["masculino", "femenino"], "Sexo inválido"
    assert nivel_actividad.lower() in ACTIVITY_FACTORS, f"Nivel de actividad inválido: {nivel_actividad}"
    assert meta.lower() in MACRO_GOALS, f"Meta inválida: {meta}"

    bmr               = calcular_bmr(peso_kg, altura_cm, edad, sexo)
    tdee              = calcular_tdee(bmr, nivel_actividad)
    calorias_objetivo = calcular_calorias_objetivo(tdee, meta)
    macros            = calcular_macros(calorias_objetivo, meta)

    return {
        "bmr":               bmr,
        "tdee":              tdee,
        "calorias_objetivo": calorias_objetivo,
        "meta":              meta,
        "nivel_actividad":   nivel_actividad,
        **macros,
    }


# ─────────────────────────────────────────────
# PRUEBA RÁPIDA (ejecuta: python modules/profile.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("   PRUEBA DEL MÓDULO: profile.py")
    print("=" * 50)

    casos = [
        # (peso, altura, edad, sexo, actividad, meta)
        (70,  175, 25, "masculino", "moderado",   "perder peso"),
        (60,  160, 30, "femenino",  "ligero",      "mantenerse"),
        (80,  180, 22, "masculino", "muy activo",  "ganar músculo"),
    ]

    for peso, altura, edad, sexo, actividad, meta in casos:
        perfil = calcular_perfil(peso, altura, edad, sexo, actividad, meta)
        print(f"\n👤 {sexo.capitalize()} | {edad} años | {peso}kg | {altura}cm")
        print(f"   Actividad : {actividad}")
        print(f"   Meta      : {meta}")
        print(f"   BMR       : {perfil['bmr']} kcal/día")
        print(f"   TDEE      : {perfil['tdee']} kcal/día")
        print(f"   🎯 Objetivo: {perfil['calorias_objetivo']} kcal/día")
        print(f"   Proteína  : {perfil['protein_g']}g")
        print(f"   Carbos    : {perfil['carbs_g']}g")
        print(f"   Grasa     : {perfil['fat_g']}g")
        print("-" * 50)