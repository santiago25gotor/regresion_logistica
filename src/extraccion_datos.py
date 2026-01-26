import http.client
import json
import pandas as pd
from datetime import datetime
import time

class AmazonDataExtractor:
    
    def __init__(self):
        self.api_key = "d8f292cc75msh67e461692b8611fp13fc44jsn71e4bce35da2"
        self.api_host = "real-time-amazon-data.p.rapidapi.com"
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.api_host
        }
    
    def buscar_producto_por_nombre(self, nombre_producto, pais="US"):
        
        print(f"\n Buscando producto: '{nombre_producto}'...")
        
        try:
            conn = http.client.HTTPSConnection(self.api_host)
            
            # Endpoint de búsqueda
            query = nombre_producto.replace(" ", "%20")
            endpoint = f"/search?query={query}&page=1&country={pais}&sort_by=RELEVANCE"
            
            conn.request("GET", endpoint, headers=self.headers)
            res = conn.getresponse()
            data = res.read()
            
            resultado = json.loads(data.decode("utf-8"))
            
            if 'data' in resultado and 'products' in resultado['data']:
                productos = resultado['data']['products']
                
                print(f"\nSe encontraron {len(productos)} productos relacionados:")
                print("\n" + "="*80)
                
                for i, prod in enumerate(productos[:5], 1):
                    print(f"{i}. {prod.get('product_title', 'Sin título')[:60]}...")
                    print(f"   ASIN: {prod.get('asin', 'N/A')}")
                    print(f"   Precio: {prod.get('product_price', 'N/A')}")
                    print(f"   Rating: {prod.get('product_star_rating', 'N/A')} ⭐")
                    print("-"*80)
                
                return productos
            else:
                print("❌ No se encontraron productos")
                return None
                
        except Exception as e:
            print(f"❌ Error en la búsqueda: {e}")
            return None
        finally:
            conn.close()
    
    def obtener_detalles_producto(self, asin, pais="US"):
       
        print(f"\n Obteniendo detalles del producto {asin}...")
        
        try:
            conn = http.client.HTTPSConnection(self.api_host)
            
            endpoint = f"/product-details?asin={asin}&country={pais}"
            
            conn.request("GET", endpoint, headers=self.headers)
            res = conn.getresponse()
            data = res.read()
            
            producto = json.loads(data.decode("utf-8"))
            
            if 'data' in producto:
                print(" Detalles del producto obtenidos correctamente")
                return producto['data']
            else:
                print(" No se pudieron obtener los detalles")
                return None
                
        except Exception as e:
            print(f" Error obteniendo detalles: {e}")
            return None
        finally:
            conn.close()
    
    def obtener_reviews(self, asin, pais="US", num_paginas=3):
        """
        Obtiene las reviews de un producto
        """
        print(f"\n Obteniendo reviews del producto {asin}...")
        
        todas_reviews = []
        
        try:
            conn = http.client.HTTPSConnection(self.api_host)
            
            for pagina in range(1, num_paginas + 1):
                print(f"   Página {pagina}/{num_paginas}...", end=" ")
                
                endpoint = f"/product-reviews?asin={asin}&country={pais}&page={pagina}"
                
                conn.request("GET", endpoint, headers=self.headers)
                res = conn.getresponse()
                data = res.read()
                
                resultado = json.loads(data.decode("utf-8"))
                
                if 'data' in resultado and 'reviews' in resultado['data']:
                    reviews = resultado['data']['reviews']
                    todas_reviews.extend(reviews)
                    print(f"✅ {len(reviews)} reviews")
                else:
                    print(" Sin reviews")
                    break
                
                time.sleep(1)  
            
            print(f"\n Total de reviews obtenidas: {len(todas_reviews)}")
            return todas_reviews
            
        except Exception as e:
            print(f" Error obteniendo reviews: {e}")
            return todas_reviews
        finally:
            conn.close()
    
    def analizar_reviews(self, reviews):
        
        if not reviews:
            return None
        
        print("\n📊 Analizando reviews...")
        
        # Extraer ratings
        ratings = []
        reviews_verificadas = 0
        reviews_con_imagenes = 0
        helpful_votes_total = 0
        
        for review in reviews:
            if 'review_star_rating' in review:
                try:
                    rating = float(review['review_star_rating'].split()[0])
                    ratings.append(rating)
                except:
                    pass
            
            if review.get('is_verified_purchase'):
                reviews_verificadas += 1
            
            if review.get('review_images'):
                reviews_con_imagenes += 1
            
            if review.get('helpful_vote_statement'):
                try:
                    votes = int(review['helpful_vote_statement'].split()[0])
                    helpful_votes_total += votes
                except:
                    pass
        
        # Calcular métricas
        metricas = {
            'total_reviews_analizadas': len(reviews),
            'rating_promedio': sum(ratings) / len(ratings) if ratings else 0,
            'rating_mediana': sorted(ratings)[len(ratings)//2] if ratings else 0,
            'reviews_5_estrellas': ratings.count(5.0),
            'reviews_4_estrellas': ratings.count(4.0),
            'reviews_3_estrellas': ratings.count(3.0),
            'reviews_2_estrellas': ratings.count(2.0),
            'reviews_1_estrella': ratings.count(1.0),
            'porcentaje_reviews_verificadas': (reviews_verificadas / len(reviews) * 100) if reviews else 0,
            'porcentaje_reviews_con_imagenes': (reviews_con_imagenes / len(reviews) * 100) if reviews else 0,
            'helpful_votes_promedio': helpful_votes_total / len(reviews) if reviews else 0,
            'ratio_reviews_positivas': ((ratings.count(5.0) + ratings.count(4.0)) / len(ratings) * 100) if ratings else 0,
            'ratio_reviews_negativas': ((ratings.count(1.0) + ratings.count(2.0)) / len(ratings) * 100) if ratings else 0
        }
        
        print(" Análisis de reviews completado")
        return metricas
    
    def extraer_datos_completos(self, asin_o_nombre, pais="US", buscar_por_nombre=False):
       
        print("\n" + "="*80)
        print(" INICIANDO EXTRACCIÓN DE DATOS DE AMAZON")
        print("="*80)
        
        if buscar_por_nombre:
            productos = self.buscar_producto_por_nombre(asin_o_nombre, pais)
            if not productos:
                return None
            
           
            asin = productos[0].get('asin')
            print(f"\n✅ Producto seleccionado - ASIN: {asin}")
        else:
            asin = asin_o_nombre
        
       
        detalles = self.obtener_detalles_producto(asin, pais)
        if not detalles:
            return None
        
        
        reviews = self.obtener_reviews(asin, pais, num_paginas=3)
        
        
        metricas_reviews = self.analizar_reviews(reviews) if reviews else {}
        
        
        datos_producto = {
           
            'asin': asin,
            'nombre_producto': detalles.get('product_title', 'N/A'),
            'marca': detalles.get('brand', 'N/A'),
            'categoria': detalles.get('category_path', [{}])[0].get('name', 'N/A') if detalles.get('category_path') else 'N/A',
            
           
            'precio': self._extraer_precio(detalles.get('product_price')),
            'precio_original': self._extraer_precio(detalles.get('product_original_price')),
            
           
            'rating_promedio': float(detalles.get('product_star_rating', 0)),
            'total_reviews': int(detalles.get('product_num_ratings', 0)),
            
            
            'disponible': detalles.get('product_availability') == 'In Stock',
            'vendedor': detalles.get('sold_by', 'N/A'),
            
           
            'fecha_extraccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        
        if metricas_reviews:
            datos_producto.update(metricas_reviews)
        
        # Calcular descuento
        if datos_producto['precio'] and datos_producto['precio_original']:
            descuento = ((datos_producto['precio_original'] - datos_producto['precio']) / 
                        datos_producto['precio_original'] * 100)
            datos_producto['descuento_porcentaje'] = round(descuento, 2)
        else:
            datos_producto['descuento_porcentaje'] = 0.0
        
        print("\n" + "="*80)
        print("✅ EXTRACCIÓN COMPLETADA")
        print("="*80)
        
        return datos_producto
    
    def _extraer_precio(self, precio_str):
       
        if not precio_str:
            return None
        
        try:
            precio_limpio = precio_str.replace('$', '').replace(',', '').strip()
            return float(precio_limpio)
        except:
            return None
    
    def guardar_datos(self, datos_producto, nombre_archivo='data/producto_amazon.csv'):
        
        import os
        
        os.makedirs('data', exist_ok=True)
        
        df = pd.DataFrame([datos_producto])
        
        df.to_csv(nombre_archivo, index=False, encoding='utf-8')
        
        print(f"\n💾 Datos guardados en: {nombre_archivo}")
        print(f"📊 Columnas guardadas: {len(df.columns)}")
        
        return df
    
    def mostrar_resumen(self, datos_producto):
        
        print("\n" + "="*80)
        print(" RESUMEN DEL PRODUCTO")
        print("="*80)
        print(f"\n Producto: {datos_producto['nombre_producto'][:60]}...")
        print(f"  ASIN: {datos_producto['asin']}")
        print(f" Marca: {datos_producto['marca']}")
        print(f" Categoría: {datos_producto['categoria']}")
        print(f"\n Precio: ${datos_producto['precio']}")
        print(f" Descuento: {datos_producto['descuento_porcentaje']}%")
        print(f"\n Rating: {datos_producto['rating_promedio']}/5.0")
        print(f" Total Reviews: {datos_producto['total_reviews']}")
        
        if 'ratio_reviews_positivas' in datos_producto:
            print(f" Reviews Positivas: {datos_producto['ratio_reviews_positivas']:.1f}%")
            print(f" Reviews Negativas: {datos_producto['ratio_reviews_negativas']:.1f}%")
        
        print(f"\n Disponible: {'Sí' if datos_producto['disponible'] else 'No'}")
        print(f" Vendedor: {datos_producto['vendedor']}")
        print("="*80)


def extraer_producto_amazon(producto, buscar_por_nombre=True, pais="US"):
    """
    Función principal para extraer datos de un producto de Amazon
    
    Args:
        producto: ASIN o nombre del producto
        buscar_por_nombre: True si es nombre, False si es ASIN
        pais: Código del país (US, ES, UK, etc.)
    
    Returns:
        DataFrame con los datos del producto
    """
    extractor = AmazonDataExtractor()
    
    # Extraer datos completos
    datos = extractor.extraer_datos_completos(
        producto, 
        pais=pais, 
        buscar_por_nombre=buscar_por_nombre
    )
    
    if datos:
        
        extractor.mostrar_resumen(datos)
        
       
        df = extractor.guardar_datos(datos)
        
        return df
    else:
        print("\n❌ No se pudieron extraer los datos del producto")
        return None



if __name__ == "__main__":
    print("\n🛒 EXTRACTOR DE DATOS DE AMAZON")
    print("="*80)
    
    
    producto = input("\n🔍 Ingresa el nombre del producto (o ASIN): ").strip()
    
   
    es_asin = input("¿Es un ASIN? (s/n): ").lower() == 's'
    
    # Extraer datos
    df = extraer_producto_amazon(
        producto, 
        buscar_por_nombre=not es_asin,
        pais="US"
    )
    
    if df is not None:
        print("\n✅ Extracción completada exitosamente!")
        print("\n📊 Primeras columnas del dataset:")
        print(df.head().T)  # Transpuesto para mejor visualización