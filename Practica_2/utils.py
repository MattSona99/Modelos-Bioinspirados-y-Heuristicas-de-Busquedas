import sys
import os

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if abspath not in sys.path:
    sys.path.append(abspath)

from Practica_1.utils import distancia_manhattan_km
import random
import numpy as np
from IPython.display import display, HTML

def construir_solucion_grasp(
    estaciones_base,
    coordenadas,
    caso_bicis,
    caso_capacidad,
    tamano_rcl=3
    ):
    """
    Fase 1 de GRASP: Construye una ruta iterativamente usando una Lista de Candidatos
    basada en una heurística de potencial (Necesidad / Distancia).
    """
    ruta_construida = []
    pendientes = set(estaciones_base)
    estacion_actual = 0
    
    while pendientes:
        candidatos = []
        lat_act, lon_act = coordenadas[estacion_actual]['lat'], coordenadas[estacion_actual]['lon']

        for cand in pendientes:
            lat_cand, lon_cand = coordenadas[cand]['lat'], coordenadas[cand]['lon']
            dist = distancia_manhattan_km(lat_act, lon_act, lat_cand, lon_cand)
            if dist == 0: dist = 0.0001
            
            # Heurística: Potencial = Necesidad / Distancia
            objetivo = caso_capacidad[cand] / 2.0
            necesidad = abs(caso_bicis[cand] - objetivo)
            potencial = necesidad / dist
            
            candidatos.append((cand, potencial))
        
        # Ordenar per potencial de mayor a menor
        candidatos.sort(key=lambda x: x[1], reverse=True)
        
        # Formar la Lista de Candidatos Restringida (RCL)
        rcl = candidatos[:tamano_rcl]
        
        # Calcular probabilidades ponderadas basadas en el potencial
        suma_potenciales = sum(c[1] for c in rcl)
        if suma_potenciales > 0:
            pesos = [c[1] / suma_potenciales for c in rcl]
        else:
            pesos = [1.0] * len(rcl) # Fallback si todos los potenciales son 0
            
        # Seleccionar aleatoriamente una estación de la RCL según sus pesos
        estacion_elegida = random.choices([c[0] for c in rcl], weights=pesos, k=1)[0]
        
        ruta_construida.append(estacion_elegida)
        pendientes.remove(estacion_elegida)
        estacion_actual = estacion_elegida
    
    return ruta_construida

import random

def aplicar_busqueda_local_primer_mejor(
    ruta_base, fobj_base, res_base,
    evaluar_ruta, funcion_objetivo,
    caso_bicis, caso_capacidad, coordenadas,
    evaluaciones_totales, historial, record_absoluto, generar_vecino_swap, **kwargs
):
    """
    Motor de Búsqueda Local (Primer Mejor) extraído para ser reutilizado
    por GRASP, ILS y VNS. Incluye el registro automático para la Caja Blanca.
    """
    ruta_actual = list(ruta_base)
    fobj_actual = fobj_base
    res_actual = res_base.copy()

    n = len(ruta_actual)
    mejorando = True

    while mejorando:
        mejorando = False
        # Nota: He cambiado range(1...) a range(0...) para explorar también el primer nodo de la ruta
        movimientos_posibles = [(i, j) for i in range(0, n - 1) for j in range(i + 1, n)]
        random.shuffle(movimientos_posibles)

        for i, j in movimientos_posibles:
            vecino = generar_vecino_swap(ruta_actual, i, j)
            res_vecino = evaluar_ruta(vecino, caso_bicis, caso_capacidad, coordenadas)
            evaluaciones_totales += 1

            kms_vec = res_vecino['kms_recorridos']
            ent_vec = res_vecino['entropia']
            fobj_vec = funcion_objetivo(kms=kms_vec, entropia=ent_vec, **kwargs)

            # Si encontramos un Primer Mejor
            if fobj_vec < fobj_actual:
                ruta_actual = vecino
                fobj_actual = fobj_vec
                res_actual['kms_recorridos'] = kms_vec
                res_actual['entropia'] = ent_vec
                mejorando = True

                # Actualizar el Récord Absoluto si hemos superado el mejor global
                if fobj_actual < record_absoluto['fobj']:
                    record_absoluto['ruta'] = list(ruta_actual)
                    record_absoluto['fobj'] = fobj_actual
                    record_absoluto['kms'] = kms_vec
                    record_absoluto['entropia'] = ent_vec

                # Registrar la CAÍDA de Explotación (Caja Blanca paso a paso)
                historial['evaluaciones'].append(evaluaciones_totales)
                historial['fobj_actual'].append(fobj_actual)
                historial['fobj'].append(record_absoluto['fobj'])
                historial['kms'].append(record_absoluto['kms'])
                historial['entropia'].append(record_absoluto['entropia'])

                break # Corte del Primer Mejor

    return ruta_actual, fobj_actual, res_actual, evaluaciones_totales


def mutacion_fuerte_sublista(ruta, tamaño=4):
    """
    Operador de Mutación Fuerte para ILS y VNS.
    Extrae una sublista de tamaño fijo de forma cíclica, la desordena y la vuelve a insertar.
    """
    nueva_ruta = list(ruta)
    n = len(nueva_ruta)
    
    # Seguridad por si la ruta es muy pequeña
    if n < tamaño:
        tamaño = n

    # Generar la posición inicial aleatoria
    pos_inicial = random.randint(0, n - 1)

    # Identificar los índices de forma cíclica (usando el módulo % n)
    indices_sublista = [(pos_inicial + i) % n for i in range(tamaño)]
    
    # Extraer los valores asignados a esos índices
    valores_extraidos = [nueva_ruta[i] for i in indices_sublista]

    # Reasignarlos aleatoriamente (Shuffle)
    random.shuffle(valores_extraidos)

    # Volver a colocar los valores mezclados en los mismos índices cíclicos
    for idx, valor_mezclado in zip(indices_sublista, valores_extraidos):
        nueva_ruta[idx] = valor_mezclado

    return nueva_ruta

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. MEDIDAS DE CAJA NEGRA (BLACK-BOX)
# =============================================================================

def calcular_metricas_caja_negra(resultados_5_ejecuciones, fobj_mejor_conocido):
    """
    Calcula la Robustez Matemática (Media, Desviación, CV) y el RPD.
    'resultados_5_ejecuciones' debe ser un diccionario: {'GRASP': [f1,f2,f3,f4,f5], 'ILS': [...], ...}
    """
    print(f"{'='*65}")
    print(f"| {'Algoritmo':<10} | {'Media':<8} | {'Std (σ)':<8} | {'CV (%)':<8} | {'RPD (%)':<8} |")
    print(f"{'-'*65}")
    
    for algoritmo, valores_fobj in resultados_5_ejecuciones.items():
        media = np.mean(valores_fobj)
        desviacion = np.std(valores_fobj)
        
        # 1.1 Coeficiente de Variación (CV) [cite: 82-83]
        cv = (desviacion / media) * 100 if media > 0 else 0
        
        # 1.2 Error Relativo Porcentual (RPD) [cite: 85-87]
        mejor_obtenido = np.min(valores_fobj)
        rpd = ((mejor_obtenido - fobj_mejor_conocido) / fobj_mejor_conocido) * 100
        
        print(f"| {algoritmo:<10} | {media:<8.2f} | {desviacion:<8.2f} | {cv:<8.2f} | {rpd:<8.2f} |")
    print(f"{'='*65}")


def graficar_boxplot_caja_negra(resultados_5_ejecuciones, nombre_caso):
    """
    Genera el Diagrama de Caja y Bigotes para evaluar dispersión y calidad [cite: 90-91].
    """
    algoritmos = list(resultados_5_ejecuciones.keys())
    datos = list(resultados_5_ejecuciones.values())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Crear el boxplot
    bplot = ax.boxplot(datos, labels=algoritmos, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', color='black'),
                       medianprops=dict(color='orange', linewidth=2))
    
    # Colores similares al ejemplo del PDF para diferenciar algoritmos
    colores = ['skyblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bplot['boxes'], colores):
        patch.set_facecolor(color)
        
    ax.set_title(f'Robustez: Dispersión en 5 Ejecuciones ({nombre_caso})', fontsize=14)
    ax.set_ylabel('Función Objetivo', fontsize=12)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. MEDIDAS DE CAJA BLANCA (WHITE-BOX)
# =============================================================================

def calcular_tasa_overlap_hamming(rutas, n_posiciones_prefijo=5):
    """
    Calcula la Tasa de Diversificación (Distancia de Hamming) en las fases tempranas [cite: 112-114].
    Compara las primeras 'n' posiciones de las 5 rutas generadas por las distintas semillas.
    """
    n_rutas = len(rutas)
    if n_rutas < 2:
        return 0.0
    
    similitudes = 0
    comparaciones_totales = 0
    
    # Comparar todas las rutas entre sí (pares únicos)
    for i in range(n_rutas):
        for j in range(i + 1, n_rutas):
            ruta_a = rutas[i][:n_posiciones_prefijo]
            ruta_b = rutas[j][:n_posiciones_prefijo]
            
            # Contar cuántas estaciones coinciden exactamente en la misma posición [cite: 118]
            coincidencias = sum(1 for a, b in zip(ruta_a, ruta_b) if a == b)
            similitudes += coincidencias
            comparaciones_totales += n_posiciones_prefijo
            
    porcentaje_overlap = (similitudes / comparaciones_totales) * 100
    
    print(f">> Tasa de Overlap en las primeras {n_posiciones_prefijo} posiciones: {porcentaje_overlap:.1f}%")
    if porcentaje_overlap > 50:
        print("   ¡Alerta! Faltan diversidad. La heurística Greedy es demasiado fuerte [cite: 120-121].")
    else:
        print("   Buena diversificación. El algoritmo explora diferentes inicios de ruta.")
        
    return porcentaje_overlap

def generar_tabla_global(diccionario_resultados):
    """
    Genera la Tabla Global de Resultados con la estructura horizontal
    solicitada en la Práctica 2 (Tabla 1.1).
    """
    html = """
    <style>
        .tabla-p2 { 
            width: 100%; 
            border-collapse: collapse; 
            font-family: Arial, sans-serif; 
            margin-top: 15px; 
            background-color: #ffffff; 
            color: #000000; 
            font-size: 0.9em;
        }
        .tabla-p2 th { 
            background-color: #e0e0e0; 
            color: #000000; 
            font-weight: bold; 
            text-align: center; 
            padding: 8px 4px; 
            border: 1px solid #444;
        }
        .tabla-p2 .borde-grueso {
            border-right: 3px solid #222;
        }
        .tabla-p2 td { 
            padding: 8px 4px; 
            text-align: center; 
            border: 1px solid #bbbbbb; 
            color: #111111;
        }
        .tabla-p2 td.algo-name { 
            font-weight: bold; 
            background-color: #f5f5f5; 
            border-right: 3px solid #222;
            border-left: 3px solid #222;
            text-align: left;
            padding-left: 10px;
        }
        .tabla-p2 tbody tr:last-child td {
            border-bottom: 3px solid #222;
        }
        .tabla-p2 thead tr:first-child th {
            border-top: 3px solid #222;
        }
    </style>
    
    <h3 style="color: inherit;">Tabla 1.1. Resultados Globales</h3>
    <table class="tabla-p2">
        <thead>
            <tr>
                <th rowspan="2" class="borde-grueso" style="border-left: 3px solid #222;">Modelo</th>
                <th colspan="4" class="borde-grueso">Caso 1</th>
                <th colspan="4" class="borde-grueso">Caso 2</th>
                <th colspan="4" class="borde-grueso">Caso 3</th>
            </tr>
            <tr>
    """
    
    # Generar los sub-encabezados repetidos para cada caso
    for i in range(3):
        html += '<th>#Ev</th><th>F OBJ</th><th>Kms</th><th class="borde-grueso">Entr</th>'
        
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    # Diccionario de mapeo: (Nombre a mostrar, [palabras clave para buscar en los resultados])
    orden_algoritmos = [
        ('Greedy', ['greedy']),
        ('P.Mejor', ['primer mejor', 'p.mejor']),
        ('GRASP', ['grasp']),
        ('ILS', ['ils']),
        ('VNS', ['vns'])
    ]
    
    casos = ['Caso 1', 'Caso 2', 'Caso 3']
    
    for nombre_display, keywords in orden_algoritmos:
        html += f'<tr><td class="algo-name">{nombre_display}</td>'
        
        # Buscar la clave correcta en el diccionario que pasaste
        clave_algo = None
        for k in diccionario_resultados.keys():
            if any(kw in k.lower() for kw in keywords):
                clave_algo = k
                break
                
        for i, caso in enumerate(casos):
            # Añadir borde derecho grueso al final de cada bloque de caso
            clase_td_final = ' class="borde-grueso"'
            
            if clave_algo and caso in diccionario_resultados[clave_algo]:
                datos = diccionario_resultados[clave_algo][caso]
                
                # Extracción adaptativa (usa ev_media si existe, si no evaluaciones normales)
                ev = datos.get('ev_media', datos.get('evaluaciones', '-'))
                if isinstance(ev, (float, np.floating)): 
                    ev = f"{ev:.1f}"
                    
                fobj_val = datos.get('score_universal', datos['fobj'])
                fobj = f"{fobj_val:.4f}"
                kms = f"{datos['kms']:.2f}"
                entr = f"{datos['entropia']:.4f}"
                
                html += f"<td>{ev}</td><td>{fobj}</td><td>{kms}</td><td{clase_td_final}>{entr}</td>"
            else:
                # Celdas vacías si no hay datos para ese algoritmo/caso
                html += f"<td>-</td><td>-</td><td>-</td><td{clase_td_final}>-</td>"
                
        html += "</tr>\n"

    html += "</tbody></table>"
    display(HTML(html))