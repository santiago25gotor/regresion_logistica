# Análisis de Churn - Telco Customer

Proyecto de análisis y modelado predictivo de churn (abandono de clientes) para una empresa de telecomunicaciones.

## 📁 Estructura del Proyecto

```
proyecto/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset original
│   └── processed/
│       └── telco_clean.csv                    # ✅ CSV limpio y listo para usar
│
├── outputs/
│   └── figures/                               # ✅ Todos los gráficos generados
│       ├── distribucion_variables_numericas.png
│       ├── matriz_correlacion.png
│       ├── analisis_bivariado_numericas.png
│       ├── bivariado_*_vs_Churn.png
│       ├── coeficientes_interpretacion.png
│       ├── distribucion_probabilidades.png
│       ├── analisis_umbrales.png
│       └── curvas_roc_pr.png
│
├── src/
│   ├── carga_datos.py           # Carga del dataset
│   ├── limpieza_datos.py        # Limpieza y preparación
│   ├── exploracion.py           # EDA (guarda gráficos en outputs/figures/)
│   ├── transformacion_datos.py  # Encoding y escalado
│   └── modelo_logistico.py      # Modelado (guarda gráficos en outputs/figures/)
│
├── main.py                      # Script principal que ejecuta todo el pipeline
└── requirements.txt             # Dependencias del proyecto
```

## 🎯 Cambios Principales en la Organización

### ✅ Antes vs Ahora

**ANTES** (problemas):
- ❌ Gráficos guardados en el directorio raíz
- ❌ CSV limpio en `data/processed/` pero mezclado con lógica poco clara
- ❌ Difícil encontrar los archivos generados

**AHORA** (mejorado):
- ✅ **Todos los gráficos** → `outputs/figures/`
- ✅ **CSV limpio** → `data/processed/telco_clean.csv`
- ✅ Estructura clara y organizada
- ✅ Fácil de encontrar y compartir resultados

## 🚀 Uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Colocar el dataset

Asegúrate de que `WA_Fn-UseC_-Telco-Customer-Churn.csv` esté en:
- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`, o
- En el mismo directorio donde ejecutas el script

### 3. Ejecutar el pipeline completo

```bash
python main.py
```

Esto ejecutará automáticamente:
1. ✅ Carga de datos
2. ✅ Limpieza y preparación
3. ✅ Análisis exploratorio (EDA)
4. ✅ Transformación de datos
5. ✅ Entrenamiento del modelo
6. ✅ Evaluación e interpretación

## 📊 Archivos Generados

### CSV Limpio
- **Ubicación**: `data/processed/telco_clean.csv`
- **Contenido**: Dataset limpio y listo para usar
- **Uso**: Este es el archivo que deberías usar para análisis posteriores

### Gráficos de EDA
- **Ubicación**: `outputs/figures/`
- **Gráficos**:
  - `distribucion_variables_numericas.png`: Distribuciones de variables numéricas
  - `matriz_correlacion.png`: Correlaciones entre variables
  - `analisis_bivariado_numericas.png`: Variables numéricas vs Churn
  - `bivariado_*_vs_Churn.png`: Variables categóricas vs Churn

### Gráficos del Modelo
- **Ubicación**: `outputs/figures/`
- **Gráficos**:
  - `coeficientes_interpretacion.png`: Coeficientes y Odds Ratios
  - `distribucion_probabilidades.png`: Probabilidades predichas
  - `analisis_umbrales.png`: Impacto de diferentes umbrales
  - `curvas_roc_pr.png`: Curvas ROC y Precision-Recall

## 🔧 Módulos

### `src/carga_datos.py`
Carga el dataset desde múltiples ubicaciones posibles.

### `src/limpieza_datos.py`
- Elimina columnas innecesarias (customerID)
- Convierte TotalCharges a numérico
- Elimina valores nulos
- Convierte target Churn (Yes/No → 1/0)

### `src/exploracion.py`
- Análisis univariante (distribuciones)
- Análisis de multicolinealidad
- Análisis bivariado (variables vs target)
- **Guarda todos los gráficos en `outputs/figures/`**

### `src/transformacion_datos.py`
- One-hot encoding para variables categóricas
- Escalado de variables numéricas
- División train/test estratificada

### `src/modelo_logistico.py`
- Entrenamiento de regresión logística
- Interpretación de coeficientes y Odds Ratios
- Análisis de umbrales de decisión
- Evaluación completa (ROC, Precision-Recall)
- **Guarda todos los gráficos en `outputs/figures/`**

## 📈 Resultados Esperados

Al ejecutar `main.py`, obtendrás:
- Un **CSV limpio** listo para usar en `data/processed/`
- **12+ gráficos** organizados en `outputs/figures/`
- **Métricas de evaluación** impresas en consola
- **Interpretación** de las variables más importantes para predecir churn

## 💡 Recomendaciones

1. **Para análisis posteriores**: Usa `data/processed/telco_clean.csv`
2. **Para presentaciones**: Usa los gráficos de `outputs/figures/`
3. **Para entender el modelo**: Revisa los coeficientes e interpretaciones en consola

## 📝 Notas

- El modelo usa `class_weight='balanced'` para manejar el desbalance de clases
- Se recomienda un umbral de decisión entre 0.35-0.45 para maximizar la detección de churn
- Los Odds Ratios facilitan la interpretación práctica del impacto de cada variable
