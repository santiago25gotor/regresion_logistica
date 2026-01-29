import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def explorar_datos(df: pd.DataFrame, target: str = "Churn"):
    print("=" * 25, "Primeras filas", "=" * 25)
    print(df.head())

    print("\n" + "=" * 25, "Dimensiones", "=" * 25)
    print(df.shape)

    print("\n" + "=" * 25, "Tipos de datos", "=" * 25)
    print(df.dtypes)

    print("\n" + "=" * 25, "Info", "=" * 25)
    df.info()

    print("\n" + "=" * 25, "Valores nulos por columna", "=" * 25)
    print(df.isnull().sum().sort_values(ascending=False).head(20))

    if target in df.columns:
        print("\n" + "=" * 25, f"Balance de clases: {target}", "=" * 25)
        vc = df[target].value_counts(dropna=False)
        print(vc)
        print("\nProporciones:")
        print((vc / len(df)).round(4))

def analisis_univariante(df: pd.DataFrame):
   
    print("\n" + "=" * 60)
    print("ANÁLISIS UNIVARIANTE DETALLADO")
    print("=" * 60)
    
    # Variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print("\n### VARIABLES NUMÉRICAS ###")
        print(df[numeric_cols].describe())
        
        # Visualización de distribuciones
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribución de {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frecuencia')
        
        # Ocultar ejes vacíos
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = FIGURES_DIR / 'distribucion_variables_numericas.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Gráfico guardado: {save_path}")
    
    # Variables categóricas
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        print("\n### VARIABLES CATEGÓRICAS ###")
        for col in cat_cols:
            print(f"\n{col}:")
            print(df[col].value_counts())
            print(f"Valores únicos: {df[col].nunique()}")

def analisis_multicolinealidad(df: pd.DataFrame):
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        print("\n" + "=" * 60)
        print("ANÁLISIS DE MULTICOLINEALIDAD")
        print("=" * 60)
        
        corr_matrix = numeric_df.corr()
        print("\nMatriz de correlación:")
        print(corr_matrix)
        
        # Visualización
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True, linewidths=1)
        plt.title('Matriz de Correlación - Variables Numéricas')
        plt.tight_layout()
        save_path = FIGURES_DIR / 'matriz_correlacion.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Gráfico guardado: {save_path}")
        
        # Identificar pares altamente correlacionados
        print("\nPares con correlación alta (|r| > 0.7):")
        alta_correlacion = False
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
                    alta_correlacion = True
        
        if not alta_correlacion:
            print("  No se detectaron correlaciones altas entre variables.")

def eda_bivariado_completo(df: pd.DataFrame, target: str = "Churn"):
    
    if target not in df.columns:
        print(f"⚠️ No existe la columna target '{target}'.")
        return
    
    print("\n" + "=" * 60)
    print("ANÁLISIS BIVARIADO: VARIABLES vs TARGET")
    print("=" * 60)
    
    # Variables numéricas vs target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    if numeric_cols:
        n_cols = len(numeric_cols)
        fig, axes = plt.subplots(n_cols, 2, figsize=(14, 5 * n_cols))
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(numeric_cols):
            # Boxplot
            df.boxplot(column=col, by=target, ax=axes[idx, 0])
            axes[idx, 0].set_title(f'{col} por {target}')
            axes[idx, 0].set_xlabel(target)
            plt.sca(axes[idx, 0])
            plt.xticks([1, 2], df[target].unique())
            
            # Histogramas superpuestos
            for label in df[target].unique():
                subset = df[df[target] == label][col].dropna()
                axes[idx, 1].hist(subset, bins=30, alpha=0.5, label=str(label))
            axes[idx, 1].set_title(f'Distribución de {col} por {target}')
            axes[idx, 1].set_xlabel(col)
            axes[idx, 1].legend()
        
        plt.tight_layout()
        save_path = FIGURES_DIR / 'analisis_bivariado_numericas.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Gráfico guardado: {save_path}")
    
    # Variables categóricas vs target
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if target in cat_cols:
        cat_cols.remove(target)
    
    if cat_cols:
        # Limitar a las 6 primeras variables categóricas más relevantes
        for col in cat_cols[:6]:
            print(f"\nTabla de contingencia: {col} vs {target}")
            ct = pd.crosstab(df[col], df[target], normalize='index')
            print(ct.round(3))
            
            # Visualización
            ct.plot(kind='bar', figsize=(10, 5))
            plt.title(f'{col} vs {target}')
            plt.xlabel(col)
            plt.ylabel('Proporción')
            plt.legend(title=target)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = FIGURES_DIR / f'bivariado_{col}_vs_{target}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"\n✓ Gráficos guardados para {min(len(cat_cols), 6)} variables categóricas en {FIGURES_DIR}")

def eda_bivariado_basico(df: pd.DataFrame, target: str = "Churn"):
    
    if target not in df.columns:
        print(f"⚠️ No existe la columna target '{target}'.")
        return

    if "tenure" in df.columns:
        plt.figure(figsize=(8, 5))
        for label in df[target].dropna().unique():
            subset = df[df[target] == label]["tenure"].dropna()
            plt.hist(subset, bins=30, alpha=0.6, label=str(label))
        plt.title("Distribución de tenure por clase (Churn)")
        plt.xlabel("tenure")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.tight_layout()
        save_path = FIGURES_DIR / 'tenure_por_churn.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    if "MonthlyCharges" in df.columns:
        plt.figure(figsize=(6, 5))
        data = []
        labels = []
        for label in df[target].dropna().unique():
            data.append(df[df[target] == label]["MonthlyCharges"].dropna())
            labels.append(str(label))
        plt.boxplot(data, labels=labels)
        plt.title("MonthlyCharges por clase (Churn)")
        plt.xlabel("Churn")
        plt.ylabel("MonthlyCharges")
        plt.tight_layout()
        save_path = FIGURES_DIR / 'monthly_charges_por_churn.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()