import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def transformar_y_dividir(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42
):
    
    if target not in df.columns:
        raise ValueError(f"No existe la columna target '{target}'.")

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # One-hot encoding para variables categóricas
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Identificar numéricas (post get_dummies ya es todo numérico, pero escalamos solo continuas originales)
    # Aquí escalamos columnas continuas típicas:
    numeric_candidates = ["tenure", "MonthlyCharges", "TotalCharges"]
    numeric_cols = [c for c in numeric_candidates if c in X_encoded.columns]

    scaler = StandardScaler()
    if numeric_cols:
        X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler