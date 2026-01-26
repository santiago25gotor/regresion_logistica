import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class TransformadorDatosAmazon:
    """
    Clase para transformar y preparar datos de Amazon para regresión logística
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.df_transformed = None
        self.scalers = {}
        self.encoders = {}
        self.features_creadas = []
    
   
    def escalar_variables(self, metodo='standard'):
        """
        Escala las variables numéricas
        
        Args:
            metodo: 'standard', 'minmax', 'robust'
        """
        print("\n" + "="*80)
        print(f"📏 1. ESCALADO DE VARIABLES (Método: {metodo})")
        print("="*80)
        
        self.df_transformed = self.df.copy()
        
        
        no_escalar = [
            'es_vendible', 'asin', 'nombre_producto', 'marca', 'categoria',
            'vendedor', 'fecha_extraccion', 'disponible', 'tiene_descuento',
            'rango_precio', 'categoria_rating', 'volumen_reviews'
        ]
        
        
        columnas_numericas = self.df_transformed.select_dtypes(include=[np.number]).columns
        columnas_escalar = [col for col in columnas_numericas if col not in no_escalar]
        
        if len(columnas_escalar) == 0:
            print("⚠️ No hay variables numéricas para escalar")
            return self.df_transformed
        
        print(f"\n--- Variables a escalar: {len(columnas_escalar)} ---")
        
        
        if metodo == 'standard':
            scaler = StandardScaler()
            nombre_scaler = "StandardScaler (media=0, std=1)"
        elif metodo == 'minmax':
            scaler = MinMaxScaler()
            nombre_scaler = "MinMaxScaler (rango 0-1)"
        elif metodo == 'robust':
            scaler = RobustScaler()
            nombre_scaler = "RobustScaler (resistente a outliers)"
        else:
            print(f"⚠️ Método '{metodo}' no reconocido. Usando 'standard'")
            scaler = StandardScaler()
            nombre_scaler = "StandardScaler"
        
        print(f"Scaler seleccionado: {nombre_scaler}\n")
        
        
        valores_escalados = scaler.fit_transform(self.df_transformed[columnas_escalar])
        
        
        columnas_escaladas = [f"{col}_scaled" for col in columnas_escalar]
        df_escalado = pd.DataFrame(
            valores_escalados,
            columns=columnas_escaladas,
            index=self.df_transformed.index
        )
        
        
        self.df_transformed = pd.concat([self.df_transformed, df_escalado], axis=1)
        
        
        self.scalers['principal'] = scaler
        self.scalers['columnas'] = columnas_escalar
        
        print(f"✅ {len(columnas_escalar)} variables escaladas correctamente")
        print(f"\nEjemplo de escalado:")
        for col in columnas_escalar[:3]:
            original = self.df_transformed[col].iloc[0]
            escalado = self.df_transformed[f"{col}_scaled"].iloc[0]
            print(f"   • {col}: {original:.2f} → {escalado:.4f}")
        
        self.features_creadas.extend(columnas_escaladas)
        
        return self.df_transformed
    
    
    def codificar_variables_categoricas(self):
        """
        Codifica variables categóricas usando Label Encoding y One-Hot Encoding
        """
        print("\n" + "="*80)
        print("🔢 2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
        print("="*80)
        
        if self.df_transformed is None:
            self.df_transformed = self.df.copy()
        
        # Variables categóricas disponibles
        variables_categoricas = [
            'marca', 'categoria', 'vendedor', 'rango_precio', 
            'categoria_rating', 'volumen_reviews'
        ]
        
        vars_disponibles = [v for v in variables_categoricas if v in self.df_transformed.columns]
        
        if len(vars_disponibles) == 0:
            print("⚠️ No hay variables categóricas para codificar")
            return self.df_transformed
        
        print(f"\n--- Variables categóricas encontradas: {len(vars_disponibles)} ---\n")
        
        for var in vars_disponibles:
            # Verificar número de categorías únicas
            n_categorias = self.df_transformed[var].nunique()
            
            print(f"📊 {var}:")
            print(f"   • Categorías únicas: {n_categorias}")
            
            if n_categorias == 1:
                print(f"   ⚠️ Solo 1 categoría, no se codifica")
                continue
            
            elif n_categorias == 2:
                # Label Encoding para binarias
                le = LabelEncoder()
                self.df_transformed[f"{var}_encoded"] = le.fit_transform(
                    self.df_transformed[var].astype(str)
                )
                self.encoders[var] = le
                print(f"   ✅ Label Encoding aplicado")
                print(f"   • Mapeo: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                self.features_creadas.append(f"{var}_encoded")
            
            elif n_categorias <= 10:
                # One-Hot Encoding para variables con pocas categorías
                dummies = pd.get_dummies(
                    self.df_transformed[var], 
                    prefix=var,
                    drop_first=True  # Evitar multicolinealidad
                )
                self.df_transformed = pd.concat([self.df_transformed, dummies], axis=1)
                print(f"   ✅ One-Hot Encoding aplicado")
                print(f"   • Columnas creadas: {len(dummies.columns)}")
                self.features_creadas.extend(dummies.columns.tolist())
            
            else:
                # Para muchas categorías, usar Label Encoding
                le = LabelEncoder()
                self.df_transformed[f"{var}_encoded"] = le.fit_transform(
                    self.df_transformed[var].astype(str)
                )
                self.encoders[var] = le
                print(f"   ✅ Label Encoding aplicado (muchas categorías)")
                self.features_creadas.append(f"{var}_encoded")
        
        print(f"\n✅ Codificación completada")
        
        return self.df_transformed
    

    
    def crear_features_adicionales(self):
        """
        Crea features adicionales específicas para el análisis de saturación
        """
        print("\n" + "="*80)
        print("⚙️ 3. FEATURE ENGINEERING ADICIONAL")
        print("="*80)
        
        if self.df_transformed is None:
            self.df_transformed = self.df.copy()
        
        df_fe = self.df_transformed.copy()
        features_nuevas = []
        
        print("\n--- Creando features avanzadas ---\n")
        
        # 1. Ratio de competitividad (precio vs rating)
        if 'precio' in df_fe.columns and 'rating_promedio' in df_fe.columns:
            df_fe['competitividad'] = df_fe['precio'] / (df_fe['rating_promedio'] + 0.1)
            print("✅ competitividad: precio / rating")
            features_nuevas.append('competitividad')
        
        # 2. Índice de popularidad (reviews * rating)
        if 'total_reviews' in df_fe.columns and 'rating_promedio' in df_fe.columns:
            df_fe['popularidad'] = np.log1p(df_fe['total_reviews']) * df_fe['rating_promedio']
            print("✅ popularidad: log(reviews) * rating")
            features_nuevas.append('popularidad')
        
        # 3. Score de calidad (rating * % positivas * confiabilidad)
        if all(col in df_fe.columns for col in ['rating_promedio', 'ratio_reviews_positivas', 'confiabilidad']):
            df_fe['score_calidad'] = (
                df_fe['rating_promedio'] * 
                (df_fe['ratio_reviews_positivas'] / 100) *
                df_fe['confiabilidad']
            )
            print("✅ score_calidad: rating × %positivas × confiabilidad")
            features_nuevas.append('score_calidad')
        
        # 4. Intensidad de competencia (saturación + precio)
        if 'indice_saturacion' in df_fe.columns and 'precio' in df_fe.columns:
            # Normalizar primero
            precio_norm = (df_fe['precio'] - df_fe['precio'].min()) / (df_fe['precio'].max() - df_fe['precio'].min() + 0.01)
            saturacion_norm = (df_fe['indice_saturacion'] - df_fe['indice_saturacion'].min()) / (df_fe['indice_saturacion'].max() - df_fe['indice_saturacion'].min() + 0.01)
            df_fe['intensidad_competencia'] = saturacion_norm * precio_norm
            print("✅ intensidad_competencia: saturación × precio normalizado")
            features_nuevas.append('intensidad_competencia')
        
        # 5. Balance review (positivas - negativas)
        if 'ratio_reviews_positivas' in df_fe.columns and 'ratio_reviews_negativas' in df_fe.columns:
            df_fe['balance_reviews'] = df_fe['ratio_reviews_positivas'] - df_fe['ratio_reviews_negativas']
            print("✅ balance_reviews: %positivas - %negativas")
            features_nuevas.append('balance_reviews')
        
        # 6. Índice de valor (rating / precio)
        if 'rating_promedio' in df_fe.columns and 'precio' in df_fe.columns:
            df_fe['indice_valor'] = df_fe['rating_promedio'] / (df_fe['precio'] + 1)
            print("✅ indice_valor: rating / precio")
            features_nuevas.append('indice_valor')
        
        # 7. Engagement score (helpful votes * reviews verificadas)
        if 'engagement' in df_fe.columns and 'confiabilidad' in df_fe.columns:
            df_fe['engagement_score'] = df_fe['engagement'] * df_fe['confiabilidad']
            print("✅ engagement_score: engagement × confiabilidad")
            features_nuevas.append('engagement_score')
        
        # 8. Riesgo de saturación (binaria)
        if 'total_reviews' in df_fe.columns:
            df_fe['alto_riesgo_saturacion'] = (df_fe['total_reviews'] > 5000).astype(int)
            print("✅ alto_riesgo_saturacion: 1 si reviews > 5000")
            features_nuevas.append('alto_riesgo_saturacion')
        
        # 9. Producto premium (binaria)
        if 'precio' in df_fe.columns:
            df_fe['es_premium'] = (df_fe['precio'] > 200).astype(int)
            print("✅ es_premium: 1 si precio > $200")
            features_nuevas.append('es_premium')
        
        # 10. Rating excelente (binaria)
        if 'rating_promedio' in df_fe.columns:
            df_fe['rating_excelente'] = (df_fe['rating_promedio'] >= 4.5).astype(int)
            print("✅ rating_excelente: 1 si rating >= 4.5")
            features_nuevas.append('rating_excelente')
        
        self.df_transformed = df_fe
        self.features_creadas.extend(features_nuevas)
        
        print(f"\n✅ {len(features_nuevas)} features adicionales creadas")
        
        return self.df_transformed
    

    
    def seleccionar_features_modelo(self):
        """
        Selecciona las features más relevantes para el modelo de regresión logística
        """
        print("\n" + "="*80)
        print("🎯 4. SELECCIÓN DE FEATURES PARA EL MODELO")
        print("="*80)
        
        if self.df_transformed is None:
            print("⚠️ Primero debes transformar los datos")
            return None
        
        # Features candidatas (escaladas)
        features_numericas_scaled = [col for col in self.df_transformed.columns if col.endswith('_scaled')]
        
        # Features categóricas codificadas
        features_categoricas_encoded = [col for col in self.df_transformed.columns if col.endswith('_encoded')]
        
        # Features one-hot
        features_onehot = [col for col in self.df_transformed.columns if 
                          any(col.startswith(prefix) for prefix in ['rango_precio_', 'categoria_rating_', 'volumen_reviews_'])]
        
        # Features adicionales creadas
        features_adicionales = [
            'competitividad', 'popularidad', 'score_calidad', 
            'intensidad_competencia', 'balance_reviews', 'indice_valor',
            'engagement_score', 'alto_riesgo_saturacion', 'es_premium', 'rating_excelente'
        ]
        features_adicionales = [f for f in features_adicionales if f in self.df_transformed.columns]
        
        # Combinar todas las features
        features_modelo = (
            features_numericas_scaled + 
            features_categoricas_encoded + 
            features_onehot + 
            features_adicionales
        )
        
        print(f"\n📊 Features seleccionadas para el modelo:")
        print(f"   • Features numéricas escaladas: {len(features_numericas_scaled)}")
        print(f"   • Features categóricas codificadas: {len(features_categoricas_encoded)}")
        print(f"   • Features One-Hot: {len(features_onehot)}")
        print(f"   • Features adicionales: {len(features_adicionales)}")
        print(f"   • TOTAL: {len(features_modelo)} features")
        
        # Verificar que existe la variable objetivo
        if 'es_vendible' not in self.df_transformed.columns:
            print("\n⚠️ Variable objetivo 'es_vendible' no encontrada")
            return None
        
        # Crear dataset para el modelo
        X = self.df_transformed[features_modelo]
        y = self.df_transformed['es_vendible']
        
        print(f"\n✅ Dataset preparado:")
        print(f"   • X shape: {X.shape}")
        print(f"   • y shape: {y.shape}")
        print(f"   • Target balance: {y.value_counts().to_dict()}")
        
        return X, y, features_modelo
    
 
    def dividir_train_test(self, X, y, test_size=0.2, random_state=42):
        """
        Divide los datos en entrenamiento y prueba
        
        NOTA: Para un solo producto, esta división no es estándar,
        pero se mantiene para consistencia con la metodología
        """
        print("\n" + "="*80)
        print("📦 5. DIVISIÓN TRAIN/TEST")
        print("="*80)
        
        print("\n⚠️  NOTA IMPORTANTE:")
        print("Para un solo producto, la división train/test no es aplicable de forma tradicional.")
        print("En este caso, usaremos los datos completos para entrenamiento y evaluación.")
        print("El modelo se evaluará en el mismo producto para interpretación.\n")
        
        # Para un solo registro, retornar los mismos datos
        if len(X) == 1:
            print(f"✅ Dataset completo usado para modelo:")
            print(f"   • Total de muestras: {len(X)}")
            print(f"   • Features: {X.shape[1]}")
            
            return X, X, y, y
        
        # Si hubiera múltiples productos (futuro), hacer split normal
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y if len(y.unique()) > 1 else None
        )
        
        print(f"✅ División completada:")
        print(f"   • Train: {X_train.shape[0]} muestras")
        print(f"   • Test: {X_test.shape[0]} muestras")
        
        return X_train, X_test, y_train, y_test
    
    # =========================================================================
    # 6. VISUALIZACIÓN DE TRANSFORMACIONES
    # =========================================================================
    
    def visualizar_transformaciones(self):
        """
        Visualiza el efecto de las transformaciones
        """
        print("\n" + "="*80)
        print("📊 6. VISUALIZACIÓN DE TRANSFORMACIONES")
        print("="*80)
        
        # Comparar distribuciones antes y después del escalado
        variables_comparar = ['precio', 'total_reviews', 'rating_promedio']
        variables_disponibles = [v for v in variables_comparar if v in self.df.columns and f"{v}_scaled" in self.df_transformed.columns]
        
        if len(variables_disponibles) == 0:
            print("⚠️ No hay variables para visualizar")
            return
        
        n_vars = len(variables_disponibles)
        fig, axes = plt.subplots(n_vars, 2, figsize=(14, n_vars * 4))
        
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        for i, var in enumerate(variables_disponibles):
            # Original
            ax1 = axes[i, 0]
            self.df[var].hist(bins=30, ax=ax1, color='skyblue', edgecolor='black')
            ax1.set_title(f'{var} (Original)', fontsize=12, fontweight='bold')
            ax1.set_xlabel(var, fontsize=10)
            ax1.set_ylabel('Frecuencia', fontsize=10)
            ax1.axvline(self.df[var].mean(), color='red', linestyle='--', linewidth=2, label='Media')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Escalada
            ax2 = axes[i, 1]
            self.df_transformed[f"{var}_scaled"].hist(bins=30, ax=ax2, color='lightcoral', edgecolor='black')
            ax2.set_title(f'{var} (Escalada)', fontsize=12, fontweight='bold')
            ax2.set_xlabel(f'{var}_scaled', fontsize=10)
            ax2.set_ylabel('Frecuencia', fontsize=10)
            ax2.axvline(self.df_transformed[f"{var}_scaled"].mean(), color='red', linestyle='--', linewidth=2, label='Media')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('graficos/comparacion_escalado.png', dpi=300, bbox_inches='tight')
        print("\n✅ Gráfico guardado: graficos/comparacion_escalado.png")
        plt.show()
    
    
    def generar_reporte_transformacion(self, X, y, features_modelo):
        """
        Genera un reporte completo de las transformaciones
        """
        print("\n" + "="*80)
        print("📋 REPORTE DE TRANSFORMACIÓN DE DATOS")
        print("="*80)
        
        print("\n1️⃣ ESCALADO")
        if 'principal' in self.scalers:
            print(f"   • Método: {type(self.scalers['principal']).__name__}")
            print(f"   • Variables escaladas: {len(self.scalers['columnas'])}")
        
        print("\n2️⃣ CODIFICACIÓN")
        print(f"   • Variables codificadas: {len(self.encoders)}")
        for var, encoder in self.encoders.items():
            print(f"     • {var}: {len(encoder.classes_)} clases")
        
        print("\n3️⃣ FEATURE ENGINEERING")
        print(f"   • Features adicionales creadas: {len(self.features_creadas)}")
        
        print("\n4️⃣ DATASET FINAL")
        print(f"   • Total de features: {X.shape[1]}")
        print(f"   • Total de muestras: {X.shape[0]}")
        print(f"   • Variable objetivo: es_vendible")
        print(f"   • Balance: {y.value_counts().to_dict()}")
        
        print("\n5️⃣ FEATURES MÁS IMPORTANTES (Top 10)")
        top_features = features_modelo[:10]
        for i, feat in enumerate(top_features, 1):
            print(f"   {i}. {feat}")
        
        print("\n✅ Datos transformados y listos para regresión logística")
    
    def guardar_datos_transformados(self, X, y, nombre_archivo='data/producto_transformado.csv'):
        """
        Guarda los datos transformados
        """
        df_final = X.copy()
        df_final['es_vendible'] = y.values
        
        df_final.to_csv(nombre_archivo, index=False, encoding='utf-8')
        print(f"\n💾 Datos transformados guardados en: {nombre_archivo}")
    
    
    def ejecutar_transformacion_completa(self, metodo_escalado='standard'):
        """
        Ejecuta todo el proceso de transformación
        """
        import os
        os.makedirs('graficos', exist_ok=True)
        
        print("\n" + "="*80)
        print("🔄 INICIANDO TRANSFORMACIÓN DE DATOS")
        print("="*80)
        
        # 1. Escalado
        self.escalar_variables(metodo=metodo_escalado)
        
        # 2. Codificación
        self.codificar_variables_categoricas()
        
        # 3. Feature Engineering
        self.crear_features_adicionales()
        
        # 4. Selección de features
        resultado = self.seleccionar_features_modelo()
        if resultado is None:
            print("❌ Error en la selección de features")
            return None
        
        X, y, features_modelo = resultado
        
        # 5. División Train/Test
        X_train, X_test, y_train, y_test = self.dividir_train_test(X, y)
        
        # 6. Visualización
        self.visualizar_transformaciones()
        
        # 7. Reporte
        self.generar_reporte_transformacion(X, y, features_modelo)
        
        # 8. Guardar
        self.guardar_datos_transformados(X, y)
        
        print("\n" + "="*80)
        print("✅ TRANSFORMACIÓN COMPLETADA")
        print("="*80)
        
        return X_train, X_test, y_train, y_test, features_modelo

def transformar_datos_amazon(df, metodo_escalado='standard'):
    transformador = TransformadorDatosAmazon(df)
    resultado = transformador.ejecutar_transformacion_completa(metodo_escalado=metodo_escalado)
    return resultado

