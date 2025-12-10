# scr/multiobjetivo.py

import random
from typing import List, Tuple
from copy import deepcopy

import numpy as np
from deap import base, creator, tools, algorithms

import scr.functions as fn  # aquí está greedy2_assignment_with_capacities


# -------------------------------------------------------------------
# 1. Individuos: vector binario de longitud nH (1 = hospital abierto)
#    Objetivos:
#      f1 = número de hospitales abiertos (p)      -> minimizar
#      f2 = Z_max + penalización por demanda no atendida -> minimizar
# -------------------------------------------------------------------

def make_random_binary_individual(nH: int, p_min: int, p_max: int) -> List[int]:
    """
    Genera un individuo binario de longitud nH:
      - Elige un número p ~ U[p_min, p_max]
      - Abre p hospitales (bits=1), el resto 0.
    """
    p_max_eff = min(p_max, nH)
    p_min_eff = min(p_min, p_max_eff)
    p = random.randint(p_min_eff, p_max_eff)
    genes = [0] * nH
    idx = random.sample(range(nH), p)
    for j in idx:
        genes[j] = 1
    return genes

def repair_individual(individual: List[int], p_min: int, p_max: int) -> List[int]:
    """
    Repara un individuo binario para que cumpla:
        p_min <= número de 1s <= p_max
    """
    nH = len(individual)
    ones = [i for i, g in enumerate(individual) if g == 1]
    zeros = [i for i, g in enumerate(individual) if g == 0]
    p = len(ones)

    # Caso límite: p_max no puede ser > nH
    p_max_eff = min(p_max, nH)
    p_min_eff = min(p_min, p_max_eff)

    # Si hay demasiados 1s, apagamos algunos
    if p > p_max_eff:
        num_to_off = p - p_max_eff
        off_indices = random.sample(ones, num_to_off)
        for i in off_indices:
            individual[i] = 0
        p = p_max_eff
        ones = [i for i, g in enumerate(individual) if g == 1]
        zeros = [i for i, g in enumerate(individual) if g == 0]

    # Si hay muy pocos 1s, encendemos algunos
    if p < p_min_eff:
        num_to_on = p_min_eff - p
        # si no hay suficientes ceros, encendemos todos los que queden
        if len(zeros) < num_to_on:
            num_to_on = len(zeros)
        on_indices = random.sample(zeros, num_to_on)
        for i in on_indices:
            individual[i] = 1

    return individual

def cx_two_point_repair(ind1, ind2, p_min: int, p_max: int):
    """
    Crossover dos puntos + reparación de número de 1s.
    """
    tools.cxTwoPoint(ind1, ind2)
    repair_individual(ind1, p_min, p_max)
    repair_individual(ind2, p_min, p_max)
    return ind1, ind2


def mut_flip_bit_repair(individual, indpb: float, p_min: int, p_max: int):
    """
    Mutación flip-bit + reparación de número de 1s.
    """
    tools.mutFlipBit(individual, indpb=indpb)
    repair_individual(individual, p_min, p_max)
    return (individual,)




def fitness_multiobjective_factory(
    D: np.ndarray,
    q: np.ndarray,
    C: np.ndarray,
    bigM: float = 1e9,
    unserved_penalty: float = 1e3,
):
    """
    Crea una función de fitness multiobjetivo:

      - Entrada: individuo binario [0/1] de longitud nH.
      - Calcula:
          open_idx = índices de hospitales abiertos.
          Asignación greedy con capacidades (greedy2_assignment_with_capacities).
      - Devuelve dos objetivos (a minimizar):
          f1 = p = número de hospitales abiertos
          f2 = Z_max + unserved_penalty * demanda_no_asignada

      Si no se abre ningún hospital, se devuelve (bigM, bigM).
    """
    V, H = D.shape

    def fitness(individual: List[int]) -> Tuple[float, float]:
        open_idx = [j for j, gene in enumerate(individual) if gene == 1]
        p = len(open_idx)

        if p == 0:
            return (bigM, bigM)

        # Asignación con tu greedy mejorado
        assign, Z, cap_left, penalty_unserved, num_unserved = fn.greedy2_assignment_with_capacities(
            D, q, C, open_idx
        )

        # Objetivo 1: número de hospitales abiertos
        f1 = float(p)

        # Objetivo 2: Z + penalización por demanda no atendida
        f2 = float(Z) + unserved_penalty * float(penalty_unserved)

        return (f1, f2)

    return fitness


def build_deap_toolbox_multiobjective(
    D: np.ndarray,
    q: np.ndarray,
    C: np.ndarray,
    p_min: int,
    p_max: int,
    seed: int = 42,
):
    """
    Construye un toolbox DEAP para el problema multiobjetivo:

      - Individuo: lista binaria (0/1) de longitud nH.
      - Fitness: 2 objetivos a minimizar (p, Z_penalizado).
      - Selección: NSGA-II (selNSGA2).
      - Cruce: cxTwoPoint.
      - Mutación: mutFlipBit.
    """
    random.seed(seed)
    np.random.seed(seed)

    V, H = D.shape

    # Tipos DEAP (evitamos redefinir si ya existen)
    if "FitnessMin2" not in creator.__dict__:
        creator.create("FitnessMin2", base.Fitness, weights=(-1.0, -1.0))
    if "IndividualBinary" not in creator.__dict__:
        creator.create("IndividualBinary", list, fitness=creator.FitnessMin2)

    toolbox = base.Toolbox()

    # Generador de individuos y población
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.IndividualBinary,
        lambda: make_random_binary_individual(H, p_min, p_max),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness multiobjetivo
    toolbox.register("evaluate", fitness_multiobjective_factory(D, q, C))


    # Operadores genéticos CON REPARACIÓN
    toolbox.register("mate", cx_two_point_repair, p_min=p_min, p_max=p_max)
    toolbox.register("mutate", mut_flip_bit_repair, indpb=1.0 / H, p_min=p_min, p_max=p_max)
    toolbox.register("select", tools.selNSGA2)


    return toolbox


def run_nsga2_multiobjective(
    toolbox,
    pop_size: int = 200,
    ngen: int = 200,
    cxpb: float = 0.9,
    mutpb: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Ejecuta un GA multiobjetivo tipo NSGA-II usando eaMuPlusLambda:

      - Población inicial de tamaño pop_size.
      - ngen generaciones.
      - Devuelve:
          pop : población final
          hof : frente de Pareto (tools.ParetoFront)
          log : logbook (si se quisiera extender más adelante)
    """
    random.seed(seed)
    np.random.seed(seed)

    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()  # almacena las soluciones no dominadas

    # Estadísticas opcionales (aquí solo guardamos medias de cada objetivo)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg_f1", lambda fits: float(np.mean([f[0] for f in fits])))
    stats.register("avg_f2", lambda fits: float(np.mean([f[1] for f in fits])))
    stats.register("min_f1", lambda fits: float(np.min([f[0] for f in fits])))
    stats.register("min_f2", lambda fits: float(np.min([f[1] for f in fits])))

    pop, log = algorithms.eaMuPlusLambda(
        population=pop,
        toolbox=toolbox,
        mu=pop_size,
        lambda_=pop_size,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    return pop, hof, log
