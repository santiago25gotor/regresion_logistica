import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class LimpiadorDatosAmazon:
  
    def __init__(self, df):
        self.df = df.copy()
        self.df_original = df.copy()
        self.cambios_realizados = []
    
    def analizar_valores_nulos(self):
        
        print("\n" + "="*80)
        print("🔍 1. ANÁLISIS DE VALORES NULOS")
        print("="*80)
        
        nulos_totales = self.df.isnull().sum().sum()
        
        if nulos_totales == 0:
            print("\nNo hay valores nulos en el dataset")
            return
        
        print(f"\n Total de valores nulos: {nulos_totales}")
        
        nulos_por_columna = self.df.isnull().sum()
        porcentaje_nulos = (nulos_por_columna / len(self.df)) * 100
        
        df_nulos = pd.DataFrame({
            'Valores Nulos': nulos_por_columna,
            '% Nulos': porcentaje_nulos
        })
        
        df_nulos = df_nulos[df_nulos['Valores Nulos'] > 0].sort_values(
            by='% Nulos', ascending=False
        )
        
        if len(df_nulos) > 0:
            print("\n--- Columnas con valores nulos ---")
            print(df_nulos)
            
            # Visualización
            plt.figure(figsize=(10, 6))
            df_nulos['% Nulos'].plot(kind='bar', color='coral', edgecolor='black')
            plt.title('Porcentaje de Valores Nulos por Columna', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Columnas', fontsize=12)
            plt.ylabel('% Nulos', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('graficos/valores_nulos.png', dpi=300, bbox_inches='tight')
            print("\n✅ Gráfico guardado: graficos/valores_nulos.png")
            plt.show()
    
    def tratar_valores_nulos(self, estrategia='media'):
        
        print(f"\n--- Tratamiento de valores nulos (estrategia: {estrategia}) ---")
        
        nulos_antes = self.df.isnull().sum().sum()
        
        if nulos_antes == 0:
            print("✅ No hay valores nulos que tratar")
            return self.df
        
        columnas_numericas = self.df.select_dtypes(include=[np.number]).columns
        columnas_con_nulos = self.df.columns[self.df.isnull().any()].tolist()
        
        for col in columnas_con_nulos:
            if col in columnas_numericas:
                if estrategia == 'media':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif estrategia == 'mediana':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif estrategia == 'cero':
                    self.df[col].fillna(0, inplace=True)
                elif estrategia == 'eliminar':
                    self.df.dropna(subset=[col], inplace=True)
            else:
               
                if estrategia == 'moda':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    self.df[col].fillna('Desconocido', inplace=True)
        
        nulos_despues = self.df.isnull().sum().sum()
        
        print(f" Valores nulos antes: {nulos_antes}")
        print(f" Valores nulos después: {nulos_despues}")
        print(f" Valores tratados: {nulos_antes - nulos_despues}")
        
        self.cambios_realizados.append(f"Valores nulos tratados con estrategia: {estrategia}")
        
        return self.df
    
    
    def detectar_outliers_visualizacion(self):
        
        print("\n" + "="*80)
        print("📊 2. DETECCIÓN DE OUTLIERS")
        print("="*80)
        
       
        variables_analizar = [
            'precio', 'total_reviews', 'rating_promedio',
            'descuento_porcentaje', 'ratio_reviews_positivas',
            'ratio_reviews_negativas', 'calidad_percibida'
        ]
        
        variables_disponibles = [v for v in variables_analizar if v in self.df.columns]
        
        if len(variables_disponibles) == 0:
            print("⚠️ No hay variables numéricas para analizar outliers")
            return
        
        print(f"\n--- Analizando {len(variables_disponibles)} variables ---")
        
        
        n_vars = len(variables_disponibles)
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_vars > 1 else [axes]
        
        for i, var in enumerate(variables_disponibles):
            ax = axes[i]
            
    
            bp = ax.boxplot([self.df[var].dropna()], vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('#FF9999')
            bp['boxes'][0].set_alpha(0.7)
            
            ax.set_title(f'Outliers en {var}', fontsize=12, fontweight='bold')
            ax.set_ylabel(var, fontsize=10)
            ax.set_xticklabels([''])
            ax.grid(axis='y', alpha=0.3)
            
            
            Q1 = self.df[var].quantile(0.25)
            Q3 = self.df[var].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[var] < limite_inferior) | 
                              (self.df[var] > limite_superior)][var]
            
            n_outliers = len(outliers)
            ax.text(0.5, 0.95, f'Outliers: {n_outliers}', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        

        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('graficos/deteccion_outliers.png', dpi=300, bbox_inches='tight')
        print("\n Gráfico guardado: graficos/deteccion_outliers.png")
        plt.show()
    
    def detectar_outliers_metodo_iqr(self, columnas=None):
       
        print("\n--- Detección de outliers (Método IQR) ---")
        
        if columnas is None:
            columnas = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            columnas = [c for c in columnas if c not in 
                       ['es_vendible', 'tiene_descuento', 'disponible']]
        
        outliers_dict = {}
        
        print("\n📊 Resumen de outliers detectados:")
        print("-" * 60)
        
        for col in columnas:
            if col not in self.df.columns:
                continue
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < limite_inferior) | 
                              (self.df[col] > limite_superior)]
            
            outliers_dict[col] = {
                'cantidad': len(outliers),
                'porcentaje': (len(outliers) / len(self.df)) * 100,
                'limite_inferior': limite_inferior,
                'limite_superior': limite_superior,
                'indices': outliers.index.tolist()
            }
            
            if len(outliers) > 0:
                print(f"{col}:")
                print(f"  • Outliers: {len(outliers)}")
                print(f"  • Límites: [{limite_inferior:.2f}, {limite_superior:.2f}]")
        
        return outliers_dict
    
    def tratar_outliers(self, columnas=None, metodo='iqr', accion='mantener'):
        
        print(f"\n--- Tratamiento de outliers (acción: {accion}) ---")
        
        if accion == 'mantener':
            print("✅ Se mantendrán los outliers (recomendado para regresión logística)")
            print("   → Los outliers pueden contener información valiosa")
            self.cambios_realizados.append("Outliers mantenidos")
            return self.df
        
        filas_antes = len(self.df)
        
        if columnas is None:
            columnas = ['precio', 'total_reviews']
        
        if metodo == 'iqr':
            for col in columnas:
                if col not in self.df.columns:
                    continue
                
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
                
                if accion == 'eliminar':
                    self.df = self.df[
                        (self.df[col] >= limite_inferior) & 
                        (self.df[col] <= limite_superior)
                    ]
                
                elif accion == 'winsorizar':
                    self.df[col] = self.df[col].clip(
                        lower=limite_inferior, 
                        upper=limite_superior
                    )
        
        filas_despues = len(self.df)
        
        if accion == 'eliminar':
            print(f"Filas eliminadas: {filas_antes - filas_despues}")
        elif accion == 'winsorizar':
            print(f"Valores winsorizados en {len(columnas)} columnas")
        
        self.cambios_realizados.append(f"Outliers tratados: {accion}")
        
        return self.df
    

    
    def validar_rangos(self):
        
        print("\n" + "="*80)
        print("✓ 3. VALIDACIÓN DE RANGOS")
        print("="*80)
        
        problemas = []
        
       
        if 'rating_promedio' in self.df.columns:
            fuera_rango = self.df[
                (self.df['rating_promedio'] < 0) | 
                (self.df['rating_promedio'] > 5)
            ]
            if len(fuera_rango) > 0:
                problemas.append(f"rating_promedio fuera de rango [0,5]: {len(fuera_rango)} registros")
        
        
        if 'precio' in self.df.columns:
            precios_negativos = self.df[self.df['precio'] < 0]
            if len(precios_negativos) > 0:
                problemas.append(f"Precios negativos: {len(precios_negativos)} registros")
        

        columnas_porcentaje = [col for col in self.df.columns if 'porcentaje' in col or 'ratio' in col]
        for col in columnas_porcentaje:
            if col in self.df.columns:
                fuera_rango = self.df[(self.df[col] < 0) | (self.df[col] > 100)]
                if len(fuera_rango) > 0:
                    problemas.append(f"{col} fuera de rango [0,100]: {len(fuera_rango)} registros")
        
        if len(problemas) == 0:
            print("\n Todos los valores están en rangos válidos")
        else:
            print("\n Problemas detectados:")
            for problema in problemas:
                print(f"   • {problema}")
    
    def validar_consistencia(self):
        
        print("\n--- Validación de consistencia ---")
        
        inconsistencias = []
        
       
        if 'precio' in self.df.columns and 'precio_original' in self.df.columns:
            inconsistente = self.df[
                (self.df['precio'] > self.df['precio_original']) & 
                (self.df['precio_original'].notna())
            ]
            if len(inconsistente) > 0:
                inconsistencias.append("Precio mayor que precio original")
        
       
        estrellas = ['reviews_5_estrellas', 'reviews_4_estrellas', 'reviews_3_estrellas',
                    'reviews_2_estrellas', 'reviews_1_estrella']
        
        if all(e in self.df.columns for e in estrellas) and 'total_reviews_analizadas' in self.df.columns:
            suma_estrellas = self.df[estrellas].sum(axis=1)
            inconsistente = self.df[suma_estrellas > self.df['total_reviews_analizadas']]
            if len(inconsistente) > 0:
                inconsistencias.append("Suma de estrellas mayor que total reviews")
        
        if len(inconsistencias) == 0:
            print(" No se detectaron inconsistencias lógicas")
        else:
            print("  Inconsistencias detectadas:")
            for inc in inconsistencias:
                print(f"   • {inc}")
    
    
    
    def generar_reporte_limpieza(self):
        """
        Genera un reporte completo del proceso de limpieza
        """
        print("\n" + "="*80)
        print(" REPORTE DE LIMPIEZA DE DATOS")
        print("="*80)
        
        print("\n CAMBIOS EN DIMENSIONES")
        print(f"   • Filas originales: {len(self.df_original)}")
        print(f"   • Filas después de limpieza: {len(self.df)}")
        print(f"   • Filas eliminadas: {len(self.df_original) - len(self.df)}")
        print(f"   • Columnas originales: {len(self.df_original.columns)}")
        print(f"   • Columnas después: {len(self.df.columns)}")
        
        print("\n VALORES NULOS")
        nulos_original = self.df_original.isnull().sum().sum()
        nulos_actual = self.df.isnull().sum().sum()
        print(f"   • Nulos originales: {nulos_original}")
        print(f"   • Nulos actuales: {nulos_actual}")
        print(f"   • Nulos tratados: {nulos_original - nulos_actual}")
        
        print("\nTRANSFORMACIONES REALIZADAS")
        if len(self.cambios_realizados) > 0:
            for i, cambio in enumerate(self.cambios_realizados, 1):
                print(f"   {i}. {cambio}")
        else:
            print("   • No se realizaron transformaciones")
        
        print("\n CALIDAD DE DATOS")
        completitud = (1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        print(f"   • Completitud: {completitud:.2f}%")
        print(f"   • Variables numéricas: {len(self.df.select_dtypes(include=[np.number]).columns)}")
        print(f"   • Variables categóricas: {len(self.df.select_dtypes(include=['object', 'category']).columns)}")
        
        print("\n Datos listos para transformación y modelado")
    
    def guardar_datos_limpios(self, nombre_archivo='data/producto_limpio.csv'):
        
        self.df.to_csv(nombre_archivo, index=False, encoding='utf-8')
        print(f"\n💾 Datos limpios guardados en: {nombre_archivo}")
    
    
    def ejecutar_limpieza_completa(self, estrategia_nulos='media', 
                                   tratar_outliers_flag=False):
        import os
        os.makedirs('graficos', exist_ok=True)
        
        print("\n" + "="*80)
        print(" INICIANDO LIMPIEZA DE DATOS")
        print("="*80)
        
        self.analizar_valores_nulos()
        self.tratar_valores_nulos(estrategia=estrategia_nulos)
        
        
        self.detectar_outliers_visualizacion()
        outliers = self.detectar_outliers_metodo_iqr()
        
        
        if tratar_outliers_flag:
            self.tratar_outliers(accion='eliminar')
        else:
            self.tratar_outliers(accion='mantener')
        
        
        self.validar_rangos()
        self.validar_consistencia()
        
        
        self.generar_reporte_limpieza()
        
       
        self.guardar_datos_limpios()
        
        print("\n" + "="*80)
        print(" LIMPIEZA DE DATOS COMPLETADA")
        print("="*80)
        
        return self.df

def limpiar_datos_amazon(df, estrategia_nulos='media', tratar_outliers=False):
    
    limpiador = LimpiadorDatosAmazon(df)
    df_limpio = limpiador.ejecutar_limpieza_completa(
        estrategia_nulos=estrategia_nulos,
        tratar_outliers_flag=tratar_outliers
    )
    return df_limpio
