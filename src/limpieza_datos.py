import pandas as pd

def limpiar_telco(df: pd.DataFrame) -> pd.DataFrame:
   
    df_clean = df.copy()

    obj_cols = df_clean.select_dtypes(include="object").columns
    for c in obj_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip()

    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    
    df_clean = df_clean.dropna()

    return df_clean

def preparar_target(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
   
    if target not in df.columns:
        raise ValueError(f"No existe la columna target '{target}' en el dataframe.")

    df_out = df.copy()

    
    valores_unicos = df_out[target].astype(str).unique()
    print(f"Valores únicos en '{target}': {valores_unicos}")
    

    df_out[target] = df_out[target].astype(str).map({"No": 0, "Yes": 1})
    
   
    if df_out[target].isnull().any():
        valores_nulos = df_out[df_out[target].isnull()][target]
        raise ValueError(
            f"El target tiene nulos después del mapeo. "
            f"Valores originales no mapeados: {valores_nulos.unique()}"
        )
    
    
    df_out[target] = df_out[target].astype(int)
    
    print(f"✓ Target '{target}' convertido correctamente: 0 (No Churn), 1 (Churn)")
    print(f"  Distribución: {df_out[target].value_counts().to_dict()}")

    return df_out