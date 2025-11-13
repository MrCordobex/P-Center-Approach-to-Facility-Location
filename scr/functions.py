import random
import math
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from deap import base, creator, tools, algorithms

def make_random_individual(nH: int, p: int) -> List[int]:
    """ Devuelve una lista ordenada de índices de hospitales abiertos de tamaño p (sin repetición). """
    ind = random.sample(range(nH), p)
    ind.sort()
    return ind

def cx_set_based(ind1, ind2, nH: int, p: int):
    """
    Cruce específico para subconjuntos (evita duplicados y mantiene tamaño p).
    Devuelve hijos del MISMO TIPO que los padres (creator.Individual).
    """
    parent1 = list(ind1)
    parent2 = list(ind2)

    set1, set2 = set(parent1), set(parent2)
    common = list(set1.intersection(set2))
    only1 = list(set1 - set2)
    only2 = list(set2 - set1)

    # Hijo 1
    child1 = common.copy()
    random.shuffle(only1)
    random.shuffle(only2)
    pool1 = common + only1 + only2
    for g in pool1:
        if g not in child1:
            child1.append(g)
        if len(child1) == p:
            break

    # Hijo 2
    child2 = common.copy()
    pool2 = common + only2 + only1
    for g in pool2:
        if g not in child2:
            child2.append(g)
        if len(child2) == p:
            break

    child1.sort()
    child2.sort()

    # *** DEVOLVER MISMO TIPO ***
    cls = type(ind1)  # normalmente creator.Individual
    return cls(child1), cls(child2)


def mut_replace_gene(individual: List[int], nH: int, p: int, indpb: float = 0.2) -> Tuple[List[int]]:
    """
    Mutación: con prob indpb, sustituye un hospital por otro no incluido.
    Garantiza tamaño p y unicidad.
    """
    current = set(individual)
    all_idx = set(range(nH))
    free = list(all_idx - current)

    for pos in range(p):
        if random.random() < indpb and free:
            new_gene = random.choice(free)
            free.remove(new_gene)
            free.append(individual[pos])
            individual[pos] = new_gene

    individual.sort()
    return (individual,)

def greedy_assignment_with_capacities(
    D: np.ndarray, q: np.ndarray, C: np.ndarray, open_idx: List[int]
) -> Tuple[np.ndarray, float, Dict[int, float], float, int]:
    """
    Asigna cada ciudad al hospital abierto más cercano respetando capacidades mediante un greedy:
      - Inicializa capacidad restante.
      - Ordena ciudades por distancia mínima al conjunto abierto (o por demanda, si prefieres).
      - Para cada ciudad, intenta asignar al hospital más cercano con capacidad suficiente.
      - Si no cabe, intenta el segundo más cercano, etc.
    Devuelve:
      assign[j] = lista de ciudades asignadas (implícito vía 'assign_of_city')
      Z = máximo d(i, j) de asignaciones realizadas
      cap_left[j] = capacidad remanente
      penalty_unserved = suma de demandas sin asignar (si quedó alguna)
      num_unserved = número de ciudades sin asignación
    """
    V, H = D.shape
    open_set = set(open_idx)

    # Capacidad restante
    cap_left = {j: float(C[j]) for j in open_set}

    # Precompute: para cada i, lista de hospitales abiertos ordenados por distancia
    nearest_open = {}
    for i in range(V):
        pairs = [(j, D[i, j]) for j in open_set]
        pairs.sort(key=lambda x: x[1])
        nearest_open[i] = pairs

    assign_of_city = -np.ones(V, dtype=int)
    Z = 0.0
    penalty_unserved = 0.0
    num_unserved = 0

    # Estrategias de ordenación de ciudades:
    #  a) Por demanda descendente (grandes primero) -> mejor usa capacidad
    #  b) Por distancia al más cercano (peores casos primero)
    # Aquí: demanda descendente
    city_order = list(range(V))
    city_order.sort(key=lambda i: q[i], reverse=True)

    for i in city_order:
        assigned = False
        for j, dij in nearest_open[i]:
            needed = q[i]
            if cap_left[j] >= needed:
                cap_left[j] -= needed
                assign_of_city[i] = j
                if dij > Z:
                    Z = dij
                assigned = True
                break

        if not assigned:
            # no hubo capacidad en ninguno -> penaliza
            penalty_unserved += q[i]
            num_unserved += 1

    return assign_of_city, Z, cap_left, penalty_unserved, num_unserved

def fitness_function_factory(D: np.ndarray, q: np.ndarray, C: np.ndarray, bigM: float = 1e6, unserved_penalty: float = 1.0):
    """
    Crea una función fitness que:
      - Calcula asignación greedy con capacidades.
      - Devuelve (Z + penalizaciones,),
      donde:
        * Z = máximo de distancias de las ciudades servidas.
        * penalización = bigM si no se abre exactamente p (no debería ocurrir con nuestra codificación),
                        + unserved_penalty * (demanda no servida).
    Ajusta unserved_penalty para endurecer o suavizar soluciones parcialmente inviables.
    """
    V, H = D.shape

    def fitness(individual: List[int]) -> Tuple[float]:
        # (opcional) sanity check
        p = len(individual)
        if len(set(individual)) != p:
            return (bigM,)  # duplicados -> inválido

        assign, Z, cap_left, penalty_unserved, num_unserved = greedy_assignment_with_capacities(D, q, C, individual)
        # Penalización por demanda no atendida (puedes multiplicar por un factor grande)
        penalty = unserved_penalty * penalty_unserved
        return (Z + penalty,)

    return fitness

def build_deap_toolbox(D: np.ndarray, q: np.ndarray, C: np.ndarray, p: int, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    V, H = D.shape

    # Crear tipos
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    #toolbox.register("clone", tools.clone)

    # Registro de generador de individuos y población
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: make_random_individual(H, p))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness
    toolbox.register("evaluate", fitness_function_factory(D, q, C, bigM=1e9, unserved_penalty=1e3))

    # Selección, cruce, mutación
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cx_set_based, nH=H, p=p)
    toolbox.register("mutate", mut_replace_gene, nH=H, p=p, indpb=0.2)

    return toolbox