import sys
import os
import random

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if abspath not in sys.path:
    sys.path.append(abspath)

from Practica_1.utils import generar_vecino_swap, generar_solucion_inicial_aleatoria
from Practica_2.utils import construir_solucion_grasp, aplicar_busqueda_local_primer_mejor, mutacion_fuerte_sublista


# ---------------------------------------------
# 1. Algoritmo GRASP
# ---------------------------------------------
def busqueda_grasp(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad, evaluar_ruta, semilla,
    max_iter_grasp=10, tamano_rcl=3, **kwargs
    ):
    """
    Algoritmo GRASP (Greedy Randomized Adaptive Search Procedure).
    Genera 10 soluciones construidas probabilísticamente y las optimiza
    con el motor común de Búsqueda Local del Primer Mejor.

    tamano_rcl=3 es el valor pedido por el PDF; valores mayores aumentan la
    diversificación a costa de potencial calidad (ver estudio comparativo).
    """
    random.seed(semilla)
    evaluaciones_totales = 0

    historial = {'evaluaciones': [], 'fobj_actual': [], 'fobj': [], 'kms': [], 'entropia': []}
    record_absoluto = {'ruta': None, 'fobj': float('inf'), 'kms': 0, 'entropia': 0}

    for iter_grasp in range(1, max_iter_grasp + 1):
        # Fase de construcción Greedy Probabilística
        ruta_actual = construir_solucion_grasp(estaciones_base, coordenadas, caso_bicis, caso_capacidad,
                                               tamano_rcl=tamano_rcl)
        res_actual = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
        evaluaciones_totales += 1
        
        fobj_actual = funcion_objetivo(kms=res_actual['kms_recorridos'], entropia=res_actual['entropia'], **kwargs)
        
        # Inicializar récord si es la primera vez
        if fobj_actual < record_absoluto['fobj']:
            record_absoluto['ruta'] = list(ruta_actual)
            record_absoluto['fobj'] = fobj_actual
            record_absoluto['kms'] = res_actual['kms_recorridos']
            record_absoluto['entropia'] = res_actual['entropia']

        # Registrar PICO DE EXPLORACIÓN
        historial['evaluaciones'].append(evaluaciones_totales)
        historial['fobj_actual'].append(fobj_actual)
        historial['fobj'].append(record_absoluto['fobj'])
        historial['kms'].append(record_absoluto['kms'])
        historial['entropia'].append(record_absoluto['entropia'])
        
        # Fase de Mejora con el motor centralizado
        _, _, _, evaluaciones_totales = aplicar_busqueda_local_primer_mejor(
            ruta_actual, fobj_actual, res_actual,
            evaluar_ruta, funcion_objetivo, caso_bicis, caso_capacidad, coordenadas,
            evaluaciones_totales, historial, record_absoluto, generar_vecino_swap, **kwargs
        )
    
    return {
        'ruta': record_absoluto['ruta'], 'fobj': record_absoluto['fobj'],
        'kms': record_absoluto['kms'], 'entropia': record_absoluto['entropia'],
        'historial': historial, 'evaluaciones': evaluaciones_totales, 'semilla': semilla
    }


# ---------------------------------------------
# 2. Algoritmo ILS (Iterated Local Search)
# ---------------------------------------------
def busqueda_ils(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad, evaluar_ruta, semilla,
    max_iter_ils=10, **kwargs
    ):
    """
    Algoritmo ILS. Genera una solución inicial aleatoria, aplica búsqueda local,
    y luego realiza un ciclo de 10 mutaciones fuertes sobre la mejor solución 
    encontrada (Criterio del Mejor), optimizando cada nueva mutación.
    """
    random.seed(semilla)
    evaluaciones_totales = 0

    # 'evals_por_bl' registra la profundidad de estancamiento (PDF §2.2):
    # cuántas evaluaciones consume cada ciclo de Búsqueda Local antes de la siguiente mutación.
    historial = {'evaluaciones': [], 'fobj_actual': [], 'fobj': [], 'kms': [], 'entropia': [],
                 'evals_por_bl': []}
    record_absoluto = {'ruta': None, 'fobj': float('inf'), 'kms': 0, 'entropia': 0}

    # 1. Solución Inicial Aleatoria
    ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
    res_actual = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
    evaluaciones_totales += 1

    fobj_actual = funcion_objetivo(kms=res_actual['kms_recorridos'], entropia=res_actual['entropia'], **kwargs)

    # Iniciar el Récord Absoluto
    record_absoluto['ruta'] = list(ruta_actual)
    record_absoluto['fobj'] = fobj_actual
    record_absoluto['kms'] = res_actual['kms_recorridos']
    record_absoluto['entropia'] = res_actual['entropia']

    # Registrar el punto de partida
    historial['evaluaciones'].append(evaluaciones_totales)
    historial['fobj_actual'].append(fobj_actual)
    historial['fobj'].append(record_absoluto['fobj'])
    historial['kms'].append(record_absoluto['kms'])
    historial['entropia'].append(record_absoluto['entropia'])

    # 2. Primera Búsqueda Local (Primer Mejor)
    evals_pre_bl = evaluaciones_totales
    _, _, _, evaluaciones_totales = aplicar_busqueda_local_primer_mejor(
        ruta_actual, fobj_actual, res_actual,
        evaluar_ruta, funcion_objetivo, caso_bicis, caso_capacidad, coordenadas,
        evaluaciones_totales, historial, record_absoluto, generar_vecino_swap, **kwargs
    )
    historial['evals_por_bl'].append(evaluaciones_totales - evals_pre_bl)

    # 3. Bucle Iterativo de Perturbación (ILS)
    for iter_ils in range(max_iter_ils):
        
        # MUTACIÓN FUERTE: Según el 'Criterio del Mejor', siempre mutamos el Récord Absoluto
        ruta_mutada = mutacion_fuerte_sublista(record_absoluto['ruta'], tamaño=4)

        # Evaluar la ruta mutada
        res_mutada = evaluar_ruta(ruta_mutada, caso_bicis, caso_capacidad, coordenadas)
        evaluaciones_totales += 1
        fobj_mutada = funcion_objetivo(kms=res_mutada['kms_recorridos'], entropia=res_mutada['entropia'], **kwargs)

        # Registrar el PICO de Exploración (La perturbación empeora el costo)
        historial['evaluaciones'].append(evaluaciones_totales)
        historial['fobj_actual'].append(fobj_mutada)
        historial['fobj'].append(record_absoluto['fobj']) # El récord sigue intacto en este instante
        historial['kms'].append(record_absoluto['kms'])
        historial['entropia'].append(record_absoluto['entropia'])

        # 4. Búsqueda Local sobre la solución mutada
        evals_pre_bl = evaluaciones_totales
        _, _, _, evaluaciones_totales = aplicar_busqueda_local_primer_mejor(
            ruta_mutada, fobj_mutada, res_mutada,
            evaluar_ruta, funcion_objetivo, caso_bicis, caso_capacidad, coordenadas,
            evaluaciones_totales, historial, record_absoluto, generar_vecino_swap, **kwargs
        )
        historial['evals_por_bl'].append(evaluaciones_totales - evals_pre_bl)

    return {
        'ruta': record_absoluto['ruta'], 'fobj': record_absoluto['fobj'],
        'kms': record_absoluto['kms'], 'entropia': record_absoluto['entropia'],
        'historial': historial, 'evaluaciones': evaluaciones_totales, 'semilla': semilla
    }

# ---------------------------------------------
# 3. Algoritmo VNS (Variable Neighborhood Search)
# ---------------------------------------------
def busqueda_vns(
    funcion_objetivo, estaciones_base, coordenadas, caso_bicis, caso_capacidad, evaluar_ruta, semilla,
    bl_max=10, k_max=4, **kwargs
    ):
    """
    Algoritmo VNS. Cambia la estructura de entorno (tamaño de mutación)
    cuando se estanca, y vuelve al entorno pequeño cuando mejora.
    """
    random.seed(semilla)
    evaluaciones_totales = 0

    # 1. Inicialización [cite: 42]
    # 'mejoras_por_k' / 'intentos_por_k' registran la profundidad de estancamiento por entorno
    # (PDF §2.2): cuántas veces cada k logra mejorar vs. cuántas veces se ha intentado.
    historial = {'evaluaciones': [], 'fobj_actual': [], 'fobj': [], 'kms': [], 'entropia': [],
                 'mejoras_por_k': {kk: 0 for kk in range(1, k_max + 1)},
                 'intentos_por_k': {kk: 0 for kk in range(1, k_max + 1)}}
    record_absoluto = {'ruta': None, 'fobj': float('inf'), 'kms': 0, 'entropia': 0}
    
    # Generar solución actual aleatoria
    ruta_actual = generar_solucion_inicial_aleatoria(estaciones_base)
    res_actual = evaluar_ruta(ruta_actual, caso_bicis, caso_capacidad, coordenadas)
    evaluaciones_totales += 1
    
    fobj_actual = funcion_objetivo(kms=res_actual['kms_recorridos'], entropia=res_actual['entropia'], **kwargs)
    
    # Iniciar récords
    record_absoluto['ruta'] = list(ruta_actual)
    record_absoluto['fobj'] = fobj_actual
    record_absoluto['kms'] = res_actual['kms_recorridos']
    record_absoluto['entropia'] = res_actual['entropia']
    
    k = 1
    bl = 0
    n = len(ruta_actual)

    # Registrar punto inicial
    historial['evaluaciones'].append(evaluaciones_totales)
    historial['fobj_actual'].append(fobj_actual)
    historial['fobj'].append(record_absoluto['fobj'])
    historial['kms'].append(record_absoluto['kms'])
    historial['entropia'].append(record_absoluto['entropia'])

    # Bucle principal de VNS [cite: 48]
    while bl < bl_max:
        
        # 2. Control de k [cite: 43]
        if k > k_max:
            k = 1
            
        # Determinar tamaño de sublista s según k [cite: 54-56]
        if k == 1: s = max(2, n // 6)
        elif k == 2: s = max(2, n // 5)
        elif k == 3: s = max(2, n // 4)
        else: s = max(2, n // 3) # k == 4
        
        # 3. Generar solución vecina (Shaking) [cite: 44-45]
        # Usamos el operador de sublista aleatoria cíclica [cite: 49]
        ruta_vecina = mutacion_fuerte_sublista(ruta_actual, tamaño=s)
        res_vecina = evaluar_ruta(ruta_vecina, caso_bicis, caso_capacidad, coordenadas)
        evaluaciones_totales += 1
        fobj_vecina = funcion_objetivo(kms=res_vecina['kms_recorridos'], entropia=res_vecina['entropia'], **kwargs)

        # Registrar PICO de Exploración (Cambio de entorno)
        historial['evaluaciones'].append(evaluaciones_totales)
        historial['fobj_actual'].append(fobj_vecina)
        historial['fobj'].append(record_absoluto['fobj'])
        historial['kms'].append(record_absoluto['kms'])
        historial['entropia'].append(record_absoluto['entropia'])

        # 4. Búsqueda Local (Primer Mejor) [cite: 46]
        # Incrementamos bl (contador de búsquedas locales realizadas)
        ruta_optimizada, fobj_optimizado, res_optimizado, evaluaciones_totales = aplicar_busqueda_local_primer_mejor(
            ruta_vecina, fobj_vecina, res_vecina,
            evaluar_ruta, funcion_objetivo, caso_bicis, caso_capacidad, coordenadas,
            evaluaciones_totales, historial, record_absoluto, generar_vecino_swap, **kwargs
        )
        bl += 1
        historial['intentos_por_k'][k] += 1

        # 5. Criterio de Aceptación y Cambio de Entorno [cite: 47]
        if fobj_optimizado < fobj_actual:
            # Mejora encontrada: actualizamos base y reseteamos entorno a k=1
            historial['mejoras_por_k'][k] += 1
            ruta_actual = list(ruta_optimizada)
            fobj_actual = fobj_optimizado
            res_actual = res_optimizado.copy()
            k = 1
        else:
            # No hay mejora: probamos con el siguiente entorno más grande
            k += 1

    return {
        'ruta': record_absoluto['ruta'], 
        'fobj': record_absoluto['fobj'],
        'kms': record_absoluto['kms'], 
        'entropia': record_absoluto['entropia'],
        'historial': historial, 
        'evaluaciones': evaluaciones_totales, 
        'semilla': semilla
    }