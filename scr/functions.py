import random
import math
from typing import List, Tuple, Dict,Iterable, Optional
from copy import deepcopy
from collections import deque
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import utils.utils as ut

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

import numpy as np
from typing import List, Tuple, Dict, Optional

def greedy2_assignment_with_capacities(
    D: np.ndarray,
    q: np.ndarray,
    C: np.ndarray,
    open_idx: List[int],
    k: int = 6,
    city_home: Optional[np.ndarray] = None,   # se ignora; se infiere el "home" con D==0
    stickiness: bool = True,                  # se ignora
    order: str = "demand_desc",               # se usa demanda descendente
    early_stop_Z: Optional[float] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, Dict[int, float], float, int]:
    """
    Greedy FRACCIONAL en 2 fases (misma firma / salidas que tu función original):

      Fase 1 (prioridad local):
        - Identifica ciudades con hospital propio ABIERTO (usando D[i,j]≈0).
        - Atiende esas ciudades en su hospital local, empezando por q desc, fraccionando si hace falta.

      Fase 2 (resto):
        - Atiende todas las ciudades (remanentes incluidos) por q desc,
          repartiendo fraccionalmente entre los k hospitales abiertos más cercanos.

    Devuelve:
      - assign_of_city: hospital global con MAYOR flujo recibido desde la ciudad (o -1 si sin atender)
      - Z: máximo D[i,j] entre pares con flujo > 0
      - cap_left: {j_global: capacidad restante} sólo para abiertos
      - penalty_unserved: suma de demanda sin atender
      - num_unserved: nº de ciudades con remanente > 0
    """
    V, H = D.shape
    if not open_idx:
        assign = -np.ones(V, dtype=int)
        return assign, 0.0, {}, float(np.sum(q)), int(np.sum(q > 0))

    # Asegurar arrays
    q = np.asarray(q, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    # Hospitales abiertos compactados 0..m-1
    open_arr = np.array(sorted(set(open_idx)), dtype=int)
    m = open_arr.size
    cap_left_open = C[open_arr].astype(float).copy()

    # Submatriz distancias a abiertos
    D_sub = D[:, open_arr]  # (V, m)

    # Preselección de k vecinos más cercanos (vectorizado)
    K = min(k, m)
    # Para fase 2
    idx_k = np.argpartition(D_sub, K-1, axis=1)[:, :K]           # (V, K)
    take_vals = np.take_along_axis(D_sub, idx_k, axis=1)
    ord_within = np.argsort(take_vals, axis=1)
    cand_pos = np.take_along_axis(idx_k, ord_within, axis=1)     # (V, K) en 0..m-1

    # Detectar hospital local (home) por D≈0 hacia un abierto
    # Si hay varios con ~0, se coge el primero
    self_tol = 1e-9
    is_self = np.isclose(D_sub, 0.0, atol=self_tol)              # (V, m)
    home_pos = np.full(V, -1, dtype=int)
    rows_with_any = np.any(is_self, axis=1)
    if np.any(rows_with_any):
        # tomar el primer True por fila
        first_true = np.argmax(is_self[rows_with_any], axis=1)
        home_pos[rows_with_any] = first_true

    # Orden de ciudades por demanda descendente (para ambas fases)
    city_order = np.argsort(-q)

    # Salidas
    best_h_pos = -np.ones(V, dtype=int)   # 0..m-1 (abiertos)
    best_flow  = np.zeros(V, dtype=float)
    Z = 0.0
    rem_city = q.copy()

    # ===== FASE 1: servir ciudades con hospital local abierto, q-desc =====
    cities_with_home = np.where(home_pos >= 0)[0]
    if cities_with_home.size > 0:
        order1 = cities_with_home[np.argsort(-q[cities_with_home])]
        for i in order1:
            rem = rem_city[i]
            if rem <= eps:
                continue
            hp = int(home_pos[i])       # pos compacta en open_arr
            cap = cap_left_open[hp]
            if cap <= eps:
                continue

            take = rem if rem <= cap else cap
            rem_city[i] = rem - take
            cap_left_open[hp] = cap - take

            if take > best_flow[i]:
                best_flow[i]  = take
                best_h_pos[i] = hp

            dij = float(D_sub[i, hp])
            if dij > Z:
                Z = dij
                if (early_stop_Z is not None) and (Z >= early_stop_Z):
                    # terminar temprano si interesa
                    penalty_unserved = float(np.sum(rem_city))
                    num_unserved = int(np.sum(rem_city > eps))
                    assign = np.where(best_h_pos >= 0, open_arr[best_h_pos], -1).astype(int)
                    cap_left = {int(open_arr[j]): float(cap_left_open[j]) for j in range(m)}
                    return assign, Z, cap_left, penalty_unserved, num_unserved

    # ===== FASE 2: servir remanente de todas las ciudades, q-desc =====
    order2 = np.argsort(-rem_city)  # por remanente
    for i in order2:
        rem = rem_city[i]
        if rem <= eps:
            continue

        # candidatos k por cercanía (incluye home si aún tiene capacidad)
        row_cand = cand_pos[i]  # (K,)
        # Recorrer candidatos en orden de distancia ascendente
        for jj in row_cand:
            cap = cap_left_open[int(jj)]
            if cap <= eps:
                continue

            take = rem if rem <= cap else cap
            rem -= take
            cap_left_open[int(jj)] = cap - take

            if take > best_flow[i]:
                best_flow[i]  = take
                best_h_pos[i] = int(jj)

            dij = float(D_sub[i, int(jj)])
            if dij > Z:
                Z = dij
                if (early_stop_Z is not None) and (Z >= early_stop_Z):
                    rem_city[i] = rem
                    penalty_unserved = float(np.sum(rem_city))
                    num_unserved = int(np.sum(rem_city > eps))
                    assign = np.where(best_h_pos >= 0, open_arr[best_h_pos], -1).astype(int)
                    cap_left = {int(open_arr[j]): float(cap_left_open[j]) for j in range(m)}
                    return assign, Z, cap_left, penalty_unserved, num_unserved

            if rem <= eps:
                break

        rem_city[i] = rem  # actualizar el remanente de la ciudad

    # Ensamblar salidas
    assign = np.where(best_h_pos >= 0, open_arr[best_h_pos], -1).astype(int)
    cap_left = {int(open_arr[j]): float(cap_left_open[j]) for j in range(m)}
    penalty_unserved = float(np.sum(rem_city))
    num_unserved = int(np.sum(rem_city > eps))
    return assign, float(Z), cap_left, penalty_unserved, num_unserved


# Crossover con conjuntos (intersección + diferencia)
def crossover_set_based(ind1, ind2, nH: int, p: int):
    """
    NUEVO crossover_set_based, usando la misma lógica que cx_set_based:
      - Trabaja con subconjuntos (listas de índices de hospitales).
      - Evita duplicados y mantiene tamaño p.
      - Sobrescribe ind1 e ind2 in-place y los devuelve.
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

    # Sobrescribimos los individuos DEAP in-place
    ind1[:] = child1
    ind2[:] = child2

    return ind1, ind2

# Mutacion 1-swap simple
def mutation_1swap(individual: List[int], nH: int, p: int, indpb: float = 0.2) -> Tuple[List[int]]:
    """
    NUEVA mutation_1swap:
      - Misma lógica que mut_replace_gene.
      - Recorre cada posición y con prob indpb hace un 'swap' por un hospital libre.
      - Mantiene tamaño p y unicidad.
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

# Búsqueda local (hill-climbing con 1-swap)

def local_search_1swap_ind(individual, toolbox, nH,
                           max_iterations=10,
                           neighbors_per_iteration=5,
                           rng=random):
    """
    Búsqueda local sencilla tipo hill-climbing usando movimientos 1-swap.
    Trabaja SOBRE un individuo DEAP.
    """
    # lo tratamos como lista normal
    current = list(individual)
    p = len(current)

    # fitness actual usando el toolbox
    current_f = toolbox.evaluate(current)[0]

    for _ in range(max_iterations):
        improved = False

        for _ in range(neighbors_per_iteration):
            pos_remove = rng.randrange(p)
            current_set = set(current)
            candidates_out = list(set(range(nH)) - current_set)
            if not candidates_out:
                continue

            h_add = rng.choice(candidates_out)

            neighbor = list(current)
            neighbor[pos_remove] = h_add
            neighbor_f = toolbox.evaluate(neighbor)[0]

            if neighbor_f < current_f:
                current = neighbor
                current_f = neighbor_f
                improved = True
                break

        if not improved:
            break

    # volcamos en el individuo DEAP
    individual[:] = current
    individual.fitness.values = (current_f,)
    return individual

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

        assign, Z, cap_left, penalty_unserved, num_unserved = greedy2_assignment_with_capacities(D, q, C, individual)
        # === NUEVO: penalización suave por "1 - entropía" de la ocupación ===
        # u_j = uso_j / C_j en [0,1];  φ(u) = 1 - H(u)/log(2), donde H(u) = -(u ln u + (1-u) ln(1-u))
        # φ(u) vale 0 en u=0.5 y 1 en u→0 o u→1. Promediaremos φ(u) y la ponderamos MUY poco.

        eps = 1e-12  # para evitar log(0)
        phis = []
        for j in individual:                      # hospitales abiertos en este individuo
            Cj = float(C[j])
            if Cj <= 0:
                continue
            cap_rem = float(cap_left.get(j, Cj))  # si no está, asumimos no usado
            used = max(0.0, Cj - cap_rem)
            u = used / Cj                         # ocupación en [0,1]

            # clamp suave para estabilidad numérica
            u = min(max(u, eps), 1.0 - eps)

            # entropía binaria y φ = 1 - H/ln 2
            H = -(u * math.log(u) + (1.0 - u) * math.log(1.0 - u))
            phi = 1.0 - (H / math.log(2.0))
            phis.append(phi)

        # Agregación: media simple (si prefieres ponderar por capacidad, ver comentario abajo)
        mean_phi = (sum(phis) / len(phis)) if phis else 0.0

        # Peso MUY pequeño para que solo actúe en empates (ajusta si hace falta)
        alpha_phi = 1e-3     # ~ "minutos" por unidad de φ en [0,1]
        pen_occ = alpha_phi * mean_phi

        # (Opcional, si quisieras ponderar por capacidad, sustituye las 3 líneas anteriores por):
        # Cs = [float(C[j]) for j in individual if float(C[j]) > 0]
        # w = [c / sum(Cs) for c in Cs] if Cs else []
        # pen_occ = alpha_phi * sum(wi * ph for wi, ph in zip(w, phis)) if w else 0.0

        # Penalización original por no atendidos (igual que tenías)
        penalty_unserved_term = unserved_penalty * penalty_unserved

        # Fitness final (misma salida: tupla de un float)
        F = Z + penalty_unserved_term + pen_occ
        return (F,)

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
    toolbox.register("mate", crossover_set_based, nH=H, p=p)
    toolbox.register("mutate", mutation_1swap, nH=H, p=p, indpb=0.2)

    return toolbox

#----------------Funciones adicionales para métodos de literatura--------------------------
def _ind_type(toolbox):
    """Devuelve el tipo (clase) del individuo registrado en el toolbox."""
    return type(toolbox.individual())


def _mk_ind(toolbox, genes: Iterable[int]):
    """Crea un individuo del tipo correcto, ordenado y sin repetidos."""
    cls = _ind_type(toolbox)
    lst = sorted(set(genes))
    return cls(lst)


def _evaluate(toolbox, ind):
    """Evalúa y escribe ind.fitness.values, devolviendo float."""
    f_tuple = toolbox.evaluate(ind)          # p.ej. (valor,)
    f = float(f_tuple[0])
    ind.fitness.values = (f,)                # <-- clave
    return f


def _neighbors_swaps(ind, nH: int, p: int, rng: random.Random, max_closed: int = None):
    """
    Genera vecinos por 'swap' 1-por-1:
      - Elige una posición abierta y la sustituye por un índice cerrado.
    Puedes limitar el nº de candidatos cerrados con max_closed para acelerar.
    """
    open_list = list(ind)
    open_set = set(open_list)
    closed = list(set(range(nH)) - open_set)
    rng.shuffle(closed)
    if max_closed is not None:
        closed = closed[:max_closed]
    for pos in range(p):
        old = open_list[pos]
        for new in closed:
            if new != old:
                child = list(open_list)
                child[pos] = new
                child.sort()
                yield child


def _init_logbook():
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "min", "avg", "max", "std"]
    return logbook


def _record_log(logbook, gen: int, fitness_values: List[float]):
    arr = np.asarray(fitness_values, dtype=float)
    m = float(np.min(arr))
    M = float(np.max(arr))
    avg = float(np.mean(arr))
    std = float(np.std(arr)) if arr.size > 1 else 0.0
    logbook.record(gen=gen, nevals=len(fitness_values), min=m, avg=avg, max=M, std=std)

