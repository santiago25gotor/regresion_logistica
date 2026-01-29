import pandas as pd

def limpiar_telco(df: pd.DataFrame) -> pd.DataFrame:
    
    df_clean = df.copy()

    # Normalizar espacios en columnas tipo object
    obj_cols = df_clean.select_dtypes(include="object").columns
    for c in obj_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip()

    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # Quitar nulos generados por TotalCharges (suele ser tenure=0)
    df_clean = df_clean.dropna()

    return df_clean

def preparar_target(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    
    if target not in df.columns:
        raise ValueError(f"No existe la columna target '{target}' en el dataframe.")

    df_out = df.copy()

    if df_out[target].dtype == "object":
        df_out[target] = df_out[target].map({"No": 0, "Yes": 1})
    else:
       
        df_out[target] = df_out[target].astype(int)

    
    if df_out[target].isnull().any():
        raise ValueError(
            "El target tiene nulos después del mapeo. Revisa valores distintos a Yes/No."
        )

    return df_out