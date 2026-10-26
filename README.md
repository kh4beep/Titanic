# Проект по задаче "Титаник"

## Инструкция по запуску:
1. Установить зависимости из requirements.txt
2. Указать актуальные пути до датасета в config.py (config.general.paths.dataset и test_dataset)
3. Запустить файл main.py

## Результаты:
- Проведен EDA
- Выполнен feature engineering
- Выполнен препроцессинг признаков
- Обучены с подбором гиперпараметров 9 моделей (LinearSVC, SVC, LogisticRegression, KNeighborsClassifier, GaussianNB, DecisionTreeClassifier, RandomForestClassifier, CatBoostClassifier, DLM)
- Построен усредняющий ансамбль с выбором моделей и весов (KNeighborsClassifier\*0.5 + CatBoost\*1 + DLM\*1)
- Метрика на Kaggle 0.7967