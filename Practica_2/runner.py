import os
import sys
import numpy as np

directorio_runner = os.path.dirname(__file__)
abspath = os.path.abspath(os.path.join(directorio_runner, '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

# Importaciones de la Práctica 1
from Practica_1.config import CASOS, SEMILLAS, TOLERANCIA
from Practica_1.utils import obtener_estaciones_a_visitar, cargar_coordenadas, evaluar_ruta, fobj_ratio

# Importaciones de la Práctica 2
from Practica_2.algorithms import busqueda_grasp, busqueda_ils, busqueda_vns
from Practica_2.utils import (
    generar_tabla_global,
    calcular_metricas_caja_negra, 
    graficar_boxplot_caja_negra, 
    calcular_tasa_overlap_hamming
)

def mostrar_tabla_practica(diccionario_resultados):
    """Llama a la generación de la tabla HTML horizontal."""
    generar_tabla_global(diccionario_resultados)

def ejecutar_analisis_cajas(caso_nombre="Caso 1", mejor_fobj_conocido=25.4):
    """
    Ejecuta internamente GRASP, ILS y VNS 5 veces para un caso específico.
    Recopila los datos en bruto para generar el Boxplot, RPD y Distancia de Hamming
    sin ensuciar el notebook.
    """
    print(f"\n{'='*80}")
    print(f" INICIANDO ANÁLISIS AVANZADO DE CAJA NEGRA Y BLANCA ({caso_nombre})")
    print(f"{'='*80}")
    
    # 1. Cargar datos del mapa
    ruta_coords = os.path.join(abspath, 'Practica_1', 'coords.json')
    coordenadas = cargar_coordenadas(ruta_coords)
    
    datos = CASOS[caso_nombre]
    bicis, capacidad = datos['bicis'], datos['capacidad']
    est_base = obtener_estaciones_a_visitar(bicis, capacidad, TOLERANCIA)
    
    # Variables para almacenar resultados en bruto de las 5 semillas
    resultados_fobj = {'GRASP': [], 'ILS': [], 'VNS': []}
    rutas_grasp = []
    
    # Usamos fobj_ratio (o la que consideres estándar) para la comparativa estadística
    funcion_objetivo = fobj_ratio 
    
    print("Ejecutando algoritmos (5 semillas)... esto puede tardar un momento.\n")
    
    for sem in SEMILLAS:
        # Extraer FOBJ de GRASP
        res_g = busqueda_grasp(funcion_objetivo, est_base, coordenadas, bicis, capacidad, evaluar_ruta, sem)
        # Convertimos a Score Universal por si internamente usó suma_ponderada
        score_g = fobj_ratio(res_g['kms'], res_g['entropia'])
        resultados_fobj['GRASP'].append(score_g)
        rutas_grasp.append(res_g['ruta'])
        
        # Extraer FOBJ de ILS
        res_i = busqueda_ils(funcion_objetivo, est_base, coordenadas, bicis, capacidad, evaluar_ruta, sem)
        score_i = fobj_ratio(res_i['kms'], res_i['entropia'])
        resultados_fobj['ILS'].append(score_i)
        
        # Extraer FOBJ de VNS
        res_v = busqueda_vns(funcion_objetivo, est_base, coordenadas, bicis, capacidad, evaluar_ruta, sem)
        score_v = fobj_ratio(res_v['kms'], res_v['entropia'])
        resultados_fobj['VNS'].append(score_v)
        
    # 2. Imprimir Métricas y Gráficas
    print("--- 1. MEDIDAS DE CAJA NEGRA ---")
    calcular_metricas_caja_negra(resultados_fobj, mejor_fobj_conocido)
    graficar_boxplot_caja_negra(resultados_fobj, caso_nombre)
    
    print("\n--- 2. MEDIDAS DE CAJA BLANCA ---")
    calcular_tasa_overlap_hamming(rutas_grasp, n_posiciones_prefijo=4)
    
    print("\nAnálisis estadístico finalizado.")