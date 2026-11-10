# Predicción de abandono de clientes (churn) – Grupo 4, 1INF24

Código del entregable final de la Tarea Académica.

## Requisitos
Python 3.10+ con `numpy`, `pandas`, `scikit-learn` y `matplotlib`.

## Datos
Descargar `WA_Fn-UseC_-Telco-Customer-Churn.csv` desde
https://www.kaggle.com/datasets/blastchar/telco-customer-churn y guardarlo como `data/telco.csv`.

## Ejecución
```
mkdir -p resultados
python churn_pipeline.py data/telco.csv resultados   # experimentos (~2 min)
python facts.py data/telco.csv resultados            # cifras descriptivas
python make_figures.py resultados                    # Figuras 2 y 3
```

## Archivos
- `churn_pipeline.py`: (A) limpieza e ingeniería de atributos, (B) partición 80/20 estratificada y validación
  cruzada de 5 pliegues, (C) codificación, estandarización y balanceo S0/S1/S2 (SMOTE-NC propio),
  (D) modelos M1 (regresión logística), M2 (CART) y M3 (híbrido árbol + RL), (E) ajuste del umbral.
- `resultados/results.json`: métricas de las 9 configuraciones (validación y prueba).
- `resultados/cv_grid.csv`: ROC-AUC de validación de cada combinación de hiperparámetros.

La semilla aleatoria es fija (42), por lo que los resultados son reproducibles.
