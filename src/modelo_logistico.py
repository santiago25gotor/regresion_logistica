import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
from pathlib import Path

# Directorio para guardar figuras
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def entrenar_logistica(X_train, y_train, max_iter: int = 2000):
    
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def interpretar_coeficientes(model, feature_names, top_n=15):
   
    print("\n" + "=" * 60)
    print("INTERPRETACIÓN DE COEFICIENTES")
    print("=" * 60)
    
    coefs = pd.DataFrame({
        'variable': feature_names,
        'coeficiente': model.coef_[0],
        'odds_ratio': np.exp(model.coef_[0]),
        'coef_abs': np.abs(model.coef_[0])
    }).sort_values('coef_abs', ascending=False)
    
    print(f"\nTop {top_n} variables más influyentes:")
    print(coefs.head(top_n).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("INTERPRETACIÓN PRÁCTICA:")
    print("=" * 60)
    
    
    for idx, row in coefs.head(3).iterrows():
        signo = "positivo" if row['coeficiente'] > 0 else "negativo"
        direccion = "aumenta" if row['coeficiente'] > 0 else "disminuye"
        odds_interp = "multiplica" if row['odds_ratio'] > 1 else "divide"
        
        print(f"\n{row['variable']}:")
        print(f"   Coeficiente: {row['coeficiente']:.4f} ({signo})")
        print(f"   Odds Ratio: {row['odds_ratio']:.4f}")
        print(f"   Interpretación: Por cada unidad que aumenta esta variable,")
        print(f"    la probabilidad de Churn {direccion} (odds se {odds_interp} por {row['odds_ratio']:.2f})")
    
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
   
    top_coefs = coefs.head(top_n)
    colors = ['red' if x < 0 else 'green' for x in top_coefs['coeficiente']]
    axes[0].barh(range(len(top_coefs)), top_coefs['coeficiente'], color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(top_coefs)))
    axes[0].set_yticklabels(top_coefs['variable'])
    axes[0].set_xlabel('Coeficiente (β)')
    axes[0].set_title(f'Top {top_n} Coeficientes más Influyentes')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0].invert_yaxis()
    
    # Gráfico 2: Odds Ratios
    axes[1].barh(range(len(top_coefs)), top_coefs['odds_ratio'], color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(top_coefs)))
    axes[1].set_yticklabels(top_coefs['variable'])
    axes[1].set_xlabel('Odds Ratio (e^β)')
    axes[1].set_title(f'Top {top_n} Odds Ratios')
    axes[1].axvline(x=1, color='black', linestyle='--', linewidth=1)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    save_path = FIGURES_DIR / 'coeficientes_interpretacion.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n Gráfico guardado: {save_path}")
    
    return coefs

def visualizar_probabilidades(model, X_test, y_test):
    
    print("\n" + "=" * 60)
    print("ANÁLISIS DE PROBABILIDADES PREDICHAS")
    print("=" * 60)
    
    probs = model.predict_proba(X_test)[:, 1]
    
   
    print("\nEstadísticas de probabilidades predichas:")
    print(f"Clase 0 (No Churn): media={probs[y_test==0].mean():.3f}, std={probs[y_test==0].std():.3f}")
    print(f"Clase 1 (Churn):    media={probs[y_test==1].mean():.3f}, std={probs[y_test==1].std():.3f}")
    
    
    plt.figure(figsize=(10, 5))
    plt.hist(probs[y_test == 0], bins=50, alpha=0.6, label='No Churn (0)', color='blue')
    plt.hist(probs[y_test == 1], bins=50, alpha=0.6, label='Churn (1)', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Umbral 0.5')
    plt.xlabel('Probabilidad predicha de Churn')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Probabilidades por Clase Real')
    plt.legend()
    plt.tight_layout()
    save_path = FIGURES_DIR / 'distribucion_probabilidades.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Gráfico guardado: {save_path}")
    
    return probs

def analizar_umbrales(y_test, probs):
   
    print("\n" + "=" * 60)
    print("ANÁLISIS DE UMBRALES DE DECISIÓN")
    print("=" * 60)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    resultados = []
    
    for th in thresholds:
        preds = (probs >= th).astype(int)
        resultados.append({
            'umbral': th,
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'f1': f1_score(y_test, preds)
        })
    
    df_umbrales = pd.DataFrame(resultados)
    print("\nMétricas por umbral:")
    print(df_umbrales.to_string(index=False))
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_umbrales['umbral'], df_umbrales['precision'], marker='o', label='Precision', linewidth=2)
    ax.plot(df_umbrales['umbral'], df_umbrales['recall'], marker='s', label='Recall', linewidth=2)
    ax.plot(df_umbrales['umbral'], df_umbrales['f1'], marker='^', label='F1-Score', linewidth=2)
    ax.set_xlabel('Umbral de Decisión')
    ax.set_ylabel('Valor de Métrica')
    ax.set_title('Impacto del Umbral en las Métricas')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = FIGURES_DIR / 'analisis_umbrales.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n Gráfico guardado: {save_path}")
    
    
    return df_umbrales

def evaluar_modelo(model, X_test, y_test, threshold: float = 0.5, show_roc: bool = True):
    
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, preds)
    auc_score = roc_auc_score(y_test, probs)

    print("\n" + "=" * 60)
    print(f"EVALUACIÓN (threshold = {threshold})")
    print("=" * 60)
    print("\nMatriz de confusión:")
    print(cm)
    print("\nInterpretación:")
    print(f"  Verdaderos Negativos (TN): {cm[0,0]} - Correctamente identificados como NO churn")
    print(f"  Falsos Positivos (FP):     {cm[0,1]} - Incorrectamente identificados como churn")
    print(f"  Falsos Negativos (FN):     {cm[1,0]} - Clientes que SÍ hicieron churn pero no detectados")
    print(f"  Verdaderos Positivos (TP): {cm[1,1]} - Correctamente identificados como churn")
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, preds, digits=4))
    print(f"AUC-ROC: {auc_score:.4f}")

    if show_roc:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
        axes[0].plot([0, 1], [0, 1], linestyle="--", color='gray', label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('Curva ROC')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Curva Precision-Recall')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = FIGURES_DIR / 'curvas_roc_pr.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"\n Gráfico guardado: {save_path}")

    return {"confusion_matrix": cm, "auc": auc_score, "probs": probs, "preds": preds}