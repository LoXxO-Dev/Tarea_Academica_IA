# Tarea_Academica_IA

Tarea Académica de Inteligencia Artificial – Grupo 4 (1INF24): predicción de abandono de clientes (churn)
en proveedores regionales de telecomunicaciones mediante modelos de clasificación interpretables.

## Contenido
- `Grupo4_TA_Churn.ipynb` – notebook ejecutado con todo el trabajo: análisis exploratorio, limpieza e ingeniería
  de atributos, partición, balanceo (S0/S1/SMOTE-NC), modelos M1–M3, ajuste del umbral, tablas y figuras del informe.
- `data/telco.csv` – conjunto de datos Telco Customer Churn de IBM
  ([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).
- `resultados/` – archivos que genera el notebook: `results.json` (métricas de las 9 configuraciones),
  `cv_grid.csv` (ROC-AUC de validación de cada combinación de hiperparámetros) y las Figuras 2 y 3.
- `documentos/Grupo4-TA-Final.pdf` y `.docx` – informe final.

## Ejecución
Requiere Python 3.10+ con `numpy`, `pandas`, `scikit-learn` y `matplotlib` (vienen instalados en Google Colab).
Abrir `Grupo4_TA_Churn.ipynb` y ejecutar todas las celdas (*Run all*); tarda entre 5 y 10 minutos.
La semilla aleatoria es fija (42) y SMOTE-NC desempata los vecinos por índice, por lo que los resultados son
idénticos en cualquier computadora.
