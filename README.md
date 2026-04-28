# 🥗 Smart Diet Planner

> AI-powered personalized meal planning system using Decision Trees, KNN, and intelligent agents — built with Python & Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 About

**Smart Diet Planner** is an intelligent system that generates personalized 7-day meal plans based on the user's physical profile, health goals, dietary restrictions, and food preferences. It combines multiple AI techniques to deliver balanced, realistic, and personalized nutrition recommendations.

This project was developed for the course **CC3045 – Inteligencia Artificial** at Universidad del Valle de Guatemala.

---

## 🧠 AI Techniques Used

| Module | Technique | Description |
|--------|-----------|-------------|
| Nutritional Profile | Linear Regression / Harris-Benedict | Calculates daily caloric needs (TDEE) |
| Food Filtering | Decision Tree | Filters foods based on allergies and restrictions |
| Recommendation | K-Nearest Neighbors (KNN) | Suggests foods similar to user preferences |
| Meal Plan Generation | Intelligent Agent / Greedy Search | Builds a balanced 7-day meal plan |
| Evaluation | Precision Metrics | Measures caloric deviation, macro coverage, diversity |

---

## 🚀 Features

- 📊 Personalized TDEE calculation based on age, weight, height, sex and activity level
- 🚫 Intelligent filtering of foods by allergies and dietary restrictions
- 🍎 Smart food recommendations using KNN on nutritional vectors
- 📅 Full 7-day meal plan (breakfast, lunch, dinner, snack)
- 📈 Nutritional charts and quality metrics
- 🛒 Auto-generated weekly shopping list
- 💻 Interactive web interface with Streamlit

---

## 🗂️ Project Structure

```
smart-diet-planner/
│
├── app/
│   └── main.py                   # Streamlit main app
│
├── modules/
│   ├── profile.py                # TDEE calculation (regression)
│   ├── filter.py                 # Food filtering (Decision Tree)
│   ├── recommender.py            # Food recommendation (KNN)
│   ├── planner.py                # Meal plan generation (Greedy Agent)
│   └── evaluator.py              # Quality metrics
│
├── data/
│   ├── raw/                      # Original downloaded dataset
│   └── processed/                # Cleaned dataset ready to use
│
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_entrenamiento_modelos.ipynb
│   └── 03_evaluacion.ipynb
│
├── docs/
│   └── Propuesta_Proyecto_Final.pdf
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jorge162017/smart-diet-planner.git
cd smart-diet-planner

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app/main.py
```

---

## 📦 Data Source

This project uses the **USDA FoodData Central** dataset — a free, open-source nutritional database maintained by the U.S. Department of Agriculture.

- 🔗 https://fdc.nal.usda.gov/download-data
- Format: CSV / JSON
- Contains: calories, protein, carbs, fat, fiber, sugars per food item

---

## 📊 Evaluation Metrics

- **Caloric deviation** — daily average deviation from target TDEE (goal: < 5%)
- **Macro coverage** — % of days meeting protein, carb and fat targets
- **Food diversity index** — number of unique foods across the week
- **Filtering precision** — % of recommended foods respecting user restrictions (goal: 100%)

---

## 🏗️ Development Roadmap

- [x] Project proposal
- [ ] Dataset exploration & cleaning
- [ ] Nutritional profile module (regression)
- [ ] Food filtering module (Decision Tree)
- [ ] Recommendation module (KNN)
- [ ] Meal plan agent (Greedy Search)
- [ ] Streamlit interface
- [ ] Evaluation & metrics
- [ ] Final report & presentation

---

## 👥 Team

| Name | GitHub |
|------|--------|
| Jorge | [@Jorge162017](https://github.com/Jorge162017) |
| Angel | [@aherrarte2019037](https://github.com/aherrarte2019037) |
| Jose | [@Abysswalkr](https://github.com/Abysswalkr) |

---

## 📄 License

This project is licensed under the MIT License.