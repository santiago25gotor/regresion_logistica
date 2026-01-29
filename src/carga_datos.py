from pathlib import Path
import pandas as pd

def cargar_datos():
    
    candidates = [
        Path("data") / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        Path("WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df

    raise FileNotFoundError(
        "No se encontró el CSV. Colócalo en 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv' "
        "o en la misma carpeta donde ejecutas el script."
    )
