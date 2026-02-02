from pathlib import Path

from src.carga_datos import cargar_datos
from src.exploracion import (
    explorar_datos, 
    analisis_univariante,
    analisis_multicolinealidad,
    eda_bivariado_completo
)
from src.limpieza_datos import limpiar_telco, preparar_target
from src.transformacion_datos import transformar_y_dividir
from src.modelo_logistico import (
    entrenar_logistica,
    interpretar_coeficientes,
    visualizar_probabilidades,
    analizar_umbrales,
    evaluar_modelo
)


DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    
   
    print("\n" + "="*60)
    print("PASO 1: CARGANDO DATOS...")
    print("="*60)
    df = cargar_datos()
    print(f" Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    
    print("\n" + "="*60)
    print("PASO 2: LIMPIEZA DE DATOS...")
    print("="*60)
    df_clean = limpiar_telco(df)
    df_ready = preparar_target(df_clean, target="Churn")
    print(f"Datos limpios: {df_ready.shape[0]} filas, {df_ready.shape[1]} columnas")
    
  
    clean_path = DATA_PROCESSED / "telco_clean.csv"
    df_ready.to_csv(clean_path, index=False)
    print(f"CSV limpio guardado en: {clean_path}")

  
    print("\n" + "="*60)
    print("PASO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)...")
    print("="*60)
    explorar_datos(df_ready, target="Churn")
    analisis_univariante(df_ready)
    analisis_multicolinealidad(df_ready)
    eda_bivariado_completo(df_ready, target="Churn")
    print(f"\n✓ Gráficos de EDA guardados en: {FIGURES_DIR}")

    
    print("\n" + "="*60)
    print("PASO 4: TRANSFORMACIÓN Y DIVISIÓN DE DATOS...")
    print("="*60)
    X_train, X_test, y_train, y_test, scaler = transformar_y_dividir(df_ready)
    print(f" Conjunto de entrenamiento: {X_train.shape[0]} filas")
    print(f" Conjunto de prueba: {X_test.shape[0]} filas")
    print(f" Número de features: {X_train.shape[1]}")

   
    print("\n" + "="*60)
    print("PASO 5: ENTRENAMIENTO DEL MODELO...")
    print("="*60)
    model = entrenar_logistica(X_train, y_train)
    print(" Modelo de regresión logística entrenado")

   
    print("\n" + "="*60)
    print("PASO 6: INTERPRETACIÓN Y EVALUACIÓN DEL MODELO...")
    print("="*60)
    interpretar_coeficientes(model, X_train.columns)
    probs = visualizar_probabilidades(model, X_test, y_test)
    analizar_umbrales(y_test, probs)
    evaluar_modelo(model, X_test, y_test)
    print(f"\n Gráficos de evaluación guardados en: {FIGURES_DIR}")

    
    print("\n" + "="*60)
    print(" PROCESO COMPLETADO EXITOSAMENTE")
    print("="*60)
    print(f"\nArchivos generados:")
    print(f"   Gráficos: {FIGURES_DIR}/")
    print(f"   CSV limpio: {clean_path}")
    print("\nRevisa los archivos generados para el análisis completo.")
    

if __name__ == "__main__":
    main()