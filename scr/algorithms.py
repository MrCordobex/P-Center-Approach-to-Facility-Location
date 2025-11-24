import random
import math
from typing import List, Tuple, Dict
from copy import deepcopy
import numpy as np
import pandas as pd
from collections import deque
from deap import base, creator, tools, algorithms
from scr.functions import _mk_ind, _evaluate, _init_logbook, _record_log, _neighbors_swaps, init_population_max_diversity
from sklearn.cluster import KMeans

from scr.functions import _mk_ind, _evaluate, _init_logbook, _record_log, _neighbors_swaps, local_search_1swap_ind


def run_eaSimple(toolbox, ngen=200, mu_pop=200, cxpb=0.8, mutpb=0.2, hof_size=5, verbose=True):
    """
    Algoritmo genético clásico 'textbook':
      - Población fija.
      - Reemplazo generacional completo.
      - Fácil y estable.
    """
    pop = toolbox.population(n=mu_pop)
    hof = tools.HallOfFame(hof_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("std", np.std)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                   stats=stats, halloffame=hof, verbose=verbose)
    return pop, hof, log

def run_eaMuPlusLambda(toolbox, ngen=200, mu=100, lambda_=200, cxpb=0.8, mutpb=0.2, hof_size=5, verbose=True):
    """
    (μ + λ): elitista. Los μ mejores entre padres+descendientes sobreviven.
    Útil cuando quieres conservar calidad y permitir presión selectiva alta.
    """
    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(hof_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("std", np.std)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox,
                                         mu=mu, lambda_=lambda_,
                                         cxpb=cxpb, mutpb=mutpb,
                                         ngen=ngen, stats=stats,
                                         halloffame=hof, verbose=verbose)
    return pop, hof, log

def run_eaMuCommaLambda(toolbox, ngen=200, mu=100, lambda_=200, cxpb=0.8, mutpb=0.2, hof_size=5, verbose=True):
    """
    (μ , λ): sin elitismo directo. Los padres NO sobreviven automáticamente.
    Promueve exploración, a veces evita estancamiento.
    """
    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(hof_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("std", np.std)

    pop, log = algorithms.eaMuCommaLambda(pop, toolbox,
                                          mu=mu, lambda_=lambda_,
                                          cxpb=cxpb, mutpb=mutpb,
                                          ngen=ngen, stats=stats,
                                          halloffame=hof, verbose=verbose)
    return pop, hof, log




#Algoritmos no genéticos
# =========================
# 1) Hill Climbing (best-improvement)
# =========================
def run_hill_climbing(toolbox,
                      nH: int,
                      p: int,
                      start=None,
                      max_iters: int = 200,
                      max_neighbors: int = 300,
                      seed: int = 0,
                      hof_size: int = 1,
                      verbose: bool = True):
    rng = random.Random(seed)
    log = _init_logbook()
    hof = tools.HallOfFame(hof_size)

    cur = toolbox.individual() if start is None else _mk_ind(toolbox, start)
    cur_f = _evaluate(toolbox, cur)
    best = deepcopy(cur)
    best_f = cur_f
    hof.update([best])
    _record_log(log, gen=0, fitness_values=[best_f])

    it = 0
    improved = True
    while improved and it < max_iters:
        it += 1
        improved = False
        cand_best = None
        cand_best_f = best_f

        # Best-improvement en vecindario
        fitness_buf = []
        for child_genes in _neighbors_swaps(best, nH, p, rng, max_closed=max_neighbors):
            child = _mk_ind(toolbox, child_genes)
            cf = _evaluate(toolbox, child)
            fitness_buf.append(cf)
            if cf < cand_best_f:
                cand_best_f = cf
                cand_best = child

        if cand_best is not None and cand_best_f < best_f:
            best, best_f = cand_best, cand_best_f
            hof.update([best])
            improved = True

        if not fitness_buf:
            fitness_buf = [best_f]
        _record_log(log, gen=it, fitness_values=[best_f])

        if verbose and it % 1 == 0:
            print(f"[HC] iter={it} best={best_f:.6f}")

    pop = [best]
    return pop, hof, log


# =========================
# 2) Simulated Annealing
# =========================
def run_simulated_annealing(toolbox,
                            nH: int,
                            p: int,
                            start=None,
                            T0: float = 1.0,
                            Tmin: float = 1e-3,
                            alpha: float = 0.95,
                            iters_per_T: int = 50,
                            seed: int = 0,
                            hof_size: int = 1,
                            verbose: bool = True):
    rng = random.Random(seed)
    log = _init_logbook()
    hof = tools.HallOfFame(hof_size)

    cur = toolbox.individual() if start is None else _mk_ind(toolbox, start)
    cur_f = _evaluate(toolbox, cur)
    best = deepcopy(cur)
    best_f = cur_f
    hof.update([best])
    _record_log(log, gen=0, fitness_values=[best_f])

    T = T0
    gen = 0
    while T > Tmin:
        gen += 1
        for _ in range(iters_per_T):
            # swap aleatorio 1-por-1
            pos = rng.randrange(p)
            open_set = set(cur)
            closed = list(set(range(nH)) - open_set)
            new_gene = rng.choice(closed)
            child = list(cur)
            child[pos] = new_gene
            child.sort()
            child = _mk_ind(toolbox, child)
            cf = _evaluate(toolbox, child)
            delta = cf - cur_f  # minimizar

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                cur, cur_f = child, cf
                if cur_f < best_f:
                    best, best_f = deepcopy(cur), cur_f
                    hof.update([best])

        _record_log(log, gen=gen, fitness_values=[best_f])
        if verbose and gen % 1 == 0:
            print(f"[SA] T={T:.4e} best={best_f:.6f}")
        T *= alpha

    pop = [best]
    return pop, hof, log


# =========================
# 3) Tabu Search
# =========================
def run_tabu_search(toolbox,
                    nH: int,
                    p: int,
                    start=None,
                    tabu_tenure: int = 10,
                    max_iters: int = 500,
                    max_neighbors: int = 200,
                    seed: int = 0,
                    hof_size: int = 1,
                    verbose: bool = True):
    rng = random.Random(seed)
    log = _init_logbook()
    hof = tools.HallOfFame(hof_size)

    cur = toolbox.individual() if start is None else _mk_ind(toolbox, start)
    cur_f = _evaluate(toolbox, cur)
    best = deepcopy(cur)
    best_f = cur_f
    hof.update([best])
    _record_log(log, gen=0, fitness_values=[best_f])

    tabu = deque(maxlen=tabu_tenure)

    for it in range(1, max_iters + 1):
        move_best = None
        move_child = None
        move_best_f = float("inf")

        neighs = list(_neighbors_swaps(cur, nH, p, rng, max_closed=max_neighbors))
        rng.shuffle(neighs)

        for child_genes in neighs:
            olds = tuple(sorted(set(cur) - set(child_genes)))
            news = tuple(sorted(set(child_genes) - set(cur)))
            mv = (olds, news)
            if mv in tabu:
                continue
            child = _mk_ind(toolbox, child_genes)
            cf = _evaluate(toolbox, child)
            if cf < move_best_f:
                move_best_f = cf
                move_best = mv
                move_child = child

        # Diversificación si todo es tabú o no hay vecinos válidos
        if move_child is None:
            pos = rng.randrange(p)
            closed = list(set(range(nH)) - set(cur))
            new_gene = rng.choice(closed)
            tmp = list(cur)
            tmp[pos] = new_gene
            tmp.sort()
            move_child = _mk_ind(toolbox, tmp)
            move_best = ((tuple([cur[pos]]),), (tuple([new_gene]),))
            move_best_f = _evaluate(toolbox, move_child)

        cur = move_child
        cur_f = move_best_f
        tabu.append(move_best)

        if cur_f < best_f:
            best, best_f = deepcopy(cur), cur_f
            hof.update([best])

        _record_log(log, gen=it, fitness_values=[best_f])
        if verbose and it % 1 == 0:
            print(f"[TS] iter={it} best={best_f:.6f}")

    pop = [best]
    return pop, hof, log


# =========================
# 4) ILS (Iterated Local Search) con HC o TS
# =========================
def run_iterated_local_search(toolbox,
                              nH: int,
                              p: int,
                              base: str = "hc",   # "hc" o "ts"
                              iters: int = 20,
                              perturb_k: int = 2,
                              seed: int = 0,
                              hof_size: int = 1,
                              verbose: bool = True):
    rng = random.Random(seed)
    log = _init_logbook()
    hof = tools.HallOfFame(hof_size)

    # punto de partida
    x = toolbox.individual()
    xf = _evaluate(toolbox, x)
    best = deepcopy(x)
    best_f = xf
    hof.update([best])
    _record_log(log, gen=0, fitness_values=[best_f])

    def _local(ind):
        if base == "hc":
            pop_l, hof_l, _ = run_hill_climbing(
                toolbox, nH, p, start=list(ind),
                max_iters=200, max_neighbors=200,
                seed=rng.randrange(1 << 30),
                hof_size=1, verbose=False
            )
        elif base == "ts":
            pop_l, hof_l, _ = run_tabu_search(
                toolbox, nH, p, start=list(ind),
                tabu_tenure=10, max_iters=250, max_neighbors=150,
                seed=rng.randrange(1 << 30),
                hof_size=1, verbose=False
            )
        else:
            raise ValueError("base debe ser 'hc' o 'ts'")
        return hof_l.items[0], _evaluate(toolbox, hof_l.items[0])

    # primera búsqueda local
    x, xf = _local(x)
    if xf < best_f:
        best, best_f = deepcopy(x), xf
        hof.update([best])
    _record_log(log, gen=1, fitness_values=[best_f])

    for it in range(2, iters + 1):
        # perturbación: k-swaps aleatorios
        y_genes = list(x)
        open_set = set(y_genes)
        closed = list(set(range(nH)) - open_set)
        rng.shuffle(closed)
        for k in range(perturb_k):
            pos = rng.randrange(p)
            y_genes[pos] = closed[k % max(1, len(closed))]
        y = _mk_ind(toolbox, y_genes)

        # búsqueda local
        x2, f2 = _local(y)
        if f2 < best_f:
            best, best_f = deepcopy(x2), f2
            hof.update([best])
        x, xf = x2, f2

        _record_log(log, gen=it, fitness_values=[best_f])
        if verbose:
            print(f"[ILS-{base}] iter={it} best={best_f:.6f}")

    pop = [best]
    return pop, hof, log


# =========================
# 5) GRASP (Greedy Randomized + Local Search)
# =========================
def run_grasp(toolbox,
              nH: int,
              p: int,
              iters: int = 50,
              rcl_size: int = 5,
              seed: int = 0,
              hof_size: int = 1,
              verbose: bool = True):
    rng = random.Random(seed)
    log = _init_logbook()
    hof = tools.HallOfFame(hof_size)

    def greedy_random_construction():
        S = []
        candidates = list(range(nH))
        rng.shuffle(candidates)
        while len(S) < p:
            # selecciona por RCL en base a fitness parcial
            scores = []
            for j in candidates:
                if j in S:
                    continue
                trial = _mk_ind(toolbox, S + [j])
                f = _evaluate(toolbox, trial)
                scores.append((f, j))
            scores.sort(key=lambda x: x[0])
            rcl = [j for _, j in scores[:min(rcl_size, len(scores))]]
            S.append(rng.choice(rcl))
        return _mk_ind(toolbox, S)

    best = None
    best_f = float("inf")

    for it in range(1, iters + 1):
        s0 = greedy_random_construction()
        # mejora local con HC
        pop_l, hof_l, _ = run_hill_climbing(
            toolbox, nH, p, start=list(s0),
            max_iters=150, max_neighbors=200,
            seed=rng.randrange(1 << 30),
            hof_size=1, verbose=False
        )
        s = hof_l.items[0]
        fs = _evaluate(toolbox, s)

        if fs < best_f:
            best, best_f = deepcopy(s), fs
            hof.update([best])

        _record_log(log, gen=it, fitness_values=[best_f])
        if verbose and it % 1 == 0:
            print(f"[GRASP] iter={it} best={best_f:.6f}")

    pop = [best]
    return pop, hof, log


# =========================
# 6) Ant Colony Optimization (ACO) para selección de p índices
#       — versión simple y didáctica —
# =========================
def run_aco_subset(toolbox,
                   nH: int,
                   p: int,
                   ants: int = 20,
                   iters: int = 50,
                   alpha: float = 1.0,
                   beta: float = 2.0,
                   rho: float = 0.5,
                   tau0: float = 0.1,
                   seed: int = 0,
                   hof_size: int = 1,
                   verbose: bool = True):
    rng = random.Random(seed)
    log = _init_logbook()
    hof = tools.HallOfFame(hof_size)

    tau = [tau0] * nH  # feromonas
    best = None
    best_f = float("inf")

    def construct_solution():
        S = []
        while len(S) < p:
            remaining = [j for j in range(nH) if j not in S]
            # Heurística basada en coste parcial (menor f -> mayor eta)
            partial_scores = []
            fvals = []
            for j in remaining:
                trial = _mk_ind(toolbox, S + [j])
                f = _evaluate(toolbox, trial)
                partial_scores.append((j, f))
                fvals.append(f)
            fmin = min(fvals) if fvals else 0.0

            probs = []
            for j, f in partial_scores:
                eta = 1.0 / (1e-9 + (f - fmin + 1.0))
                probs.append((j, (tau[j] ** alpha) * (eta ** beta)))

            Z = sum(w for _, w in probs)
            r = rng.random() * max(Z, 1e-12)
            acc = 0.0
            chosen = remaining[0]
            for j, w in probs:
                acc += w
                if acc >= r:
                    chosen = j
                    break
            S.append(chosen)

        ind = _mk_ind(toolbox, S)
        f = _evaluate(toolbox, ind)
        return ind, f

    # gen 0: registrar un arranque
    _record_log(log, gen=0, fitness_values=[best_f if best is not None else float("inf")])

    for it in range(1, iters + 1):
        batch = []
        for _a in range(ants):
            ind, f = construct_solution()
            batch.append((ind, f))
            if f < best_f:
                best, best_f = deepcopy(ind), f
                hof.update([best])

        # evaporación
        tau = [(1.0 - rho) * t for t in tau]

        # refuerzo (mejor de la iteración)
        it_best_ind, it_best_f = min(batch, key=lambda x: x[1])
        delta = 1.0 / (1e-9 + it_best_f)
        for j in it_best_ind:
            tau[j] += delta

        _record_log(log, gen=it, fitness_values=[best_f])
        if verbose and it % 1 == 0:
            print(f"[ACO] iter={it} best={best_f:.6f}")

    pop = [best]
    return pop, hof, log


# Algoritmo ACME
def run_acme(toolbox,p: int,
             ngen: int = 200,
             mu_pop: int = 200,
             cxpb: float = 0.8,
             mutpb: float = 0.2,
             hof_size: int = 5,
             k_clusters: int = 5,
             acme_period: int = 10,
             ls_iters: int = 5,
             kmeans_max_iter: int = 10,
             seed: int = 0,
             verbose: bool = True):
    """
    Algoritmo genético tipo eaSimple + módulo ACME:

    - GA generacional estándar (población fija, selección, cruce, mutación).
    - Cada `acme_period` generaciones:
        * Se hace un K-Means sobre la población (en el espacio de los genes).
        * Para cada cluster se toma una solución representante (medoide).
        * A cada representante se le aplica una búsqueda local rápida (mutaciones iteradas).
        * Si alguna de estas soluciones mejora a la mejor global, se actualiza el HallOfFame.
      IMPORTANTE: la población NO se modifica con este módulo; sólo se actualiza la memoria (hof).

    Devuelve:
        pop, hof, log   (como el resto de funciones)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Logbook y Hall of Fame
    log = _init_logbook()
    hof = tools.HallOfFame(hof_size)

    # Población inicial
    #pop = toolbox.population(n=mu_pop)
    pop = init_population_max_diversity(
        toolbox,
        nH=133,
        p=p,
        mu=mu_pop,
        candidates_per_ind=50,
        seed=seed,
    )

    # Evaluación inicial
    for ind in pop:
        _evaluate(toolbox, ind)
    hof.update(pop)

    # Registrar gen 0
    fits0 = [float(ind.fitness.values[0]) for ind in pop]
    _record_log(log, gen=0, fitness_values=fits0)
    if verbose:
        print(f"[ACME-GA] gen=0 pop_min={min(fits0):.6f} best_global={hof[0].fitness.values[0]:.6f}")

    # --- Helpers internos ---

        def _kmeans_representatives(population, k: int, max_iter: int = 10):
            """
            Aplica K-Means (sklearn) sobre la población (vectores de genes) y devuelve
            una lista de individuos REPRESENTANTES (medoides) de cada cluster.
            No se modifica la población.

            - population: lista de individuos (cada uno es una lista de índices de hospitales).
            - k: nº de clusters.
            - max_iter: nº máximo de iteraciones de KMeans.
            """
            N = len(population)
            if N == 0 or k <= 0:
                return []

            # Asumimos todos los individuos tienen misma longitud (p)
            p_dim = len(population[0])
            X = np.array([np.array(ind, dtype=float) for ind in population])  # (N, p_dim)

            k_eff = min(k, N)

            # K-Means con sklearn (centramos en velocidad y estabilidad)
            kmeans = KMeans(
                n_clusters=k_eff,
                n_init=10,          # varias inicializaciones para evitar malos mínimos
                max_iter=max_iter,
                random_state=None   # si quieres reproducibilidad: usa 'seed'
            )
            kmeans.fit(X)

            labels = kmeans.labels_              # (N,)
            centroids = kmeans.cluster_centers_  # (k_eff, p_dim)

            reps = []
            for j in range(k_eff):
                mask = (labels == j)
                if not np.any(mask):
                    continue
                idxs = np.where(mask)[0]
                X_cluster = X[idxs]                     # puntos del cluster j
                c = centroids[j][None, :]              # centroide (1, p_dim)
                d2_cluster = ((X_cluster - c) ** 2).sum(axis=1)  # distancias^2
                best_idx = idxs[int(np.argmin(d2_cluster))]
                rep = deepcopy(population[best_idx])
                reps.append(rep)

            return reps


    def _local_search(ind):
        """
        Búsqueda local rápida sobre un representante:
          - Punto de partida: clon del individuo dado.
          - Vecindario: aplicar la mutación registrada en el toolbox.
          - Se aceptan sólo mejoras (Hill Climbing simple).
        Devuelve el mejor individuo encontrado (NO modifica el original).
        """
        best = deepcopy(ind)
        if not best.fitness.valid:
            _evaluate(toolbox, best)

        best_f = float(best.fitness.values[0])

        for _ in range(ls_iters):
            cand = deepcopy(best)
            # En la búsqueda local aplicamos SIEMPRE mutación (sin mutpb)
            toolbox.mutate(cand)
            _evaluate(toolbox, cand)
            cf = float(cand.fitness.values[0])
            if cf < best_f:
                best, best_f = cand, cf

        return best

    # --- Bucle principal GA + ACME ---
    for gen in range(1, ngen + 1):
        # Selección
        selected = toolbox.select(pop, len(pop))
        # Clonamos con deepcopy, no tools.clone
        offspring = [deepcopy(ind) for ind in selected]

        # Cruce
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                # invalidar fitness
                if hasattr(child1.fitness, "values"):
                    del child1.fitness.values
                if hasattr(child2.fitness, "values"):
                    del child2.fitness.values

        # Mutación
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                if hasattr(mutant.fitness, "values"):
                    del mutant.fitness.values

        # Evaluar descendencia inválida
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            _evaluate(toolbox, ind)

        # Reemplazo generacional completo
        pop[:] = offspring

        # === MÓDULO ACME CADA acme_period GENERACIONES ===
        if acme_period > 0 and (gen % acme_period == 0):
            reps = _kmeans_representatives(pop, k_clusters, max_iter=kmeans_max_iter)

            if reps:
                # Asegurar que el hof no está vacío
                if len(hof) == 0:
                    hof.update(pop)

                current_best_val = float(hof[0].fitness.values[0])

                for rep in reps:
                    cand = _local_search(rep)
                    cand_val = float(cand.fitness.values[0])

                    if cand_val < current_best_val:
                        hof.update([cand])
                        current_best_val = float(hof[0].fitness.values[0])
                        if verbose:
                            print(f"[ACME] gen={gen} nueva mejor solución encontrada: {current_best_val:.6f}")

        # Actualizar Hall of Fame con la población (como en un GA normal)
        hof.update(pop)

        # Registrar estadísticas de la POBLACIÓN
        fits = [float(ind.fitness.values[0]) for ind in pop]
        _record_log(log, gen=gen, fitness_values=fits)

        if verbose:
            print(f"[ACME-GA] gen={gen} pop_min={min(fits):.6f} best_global={hof[0].fitness.values[0]:.6f}")
        # aplicar simulated annealing al mejor individuo de la generación
    pop_sa, hof_sa, _ = run_simulated_annealing(toolbox,
                        nH=133,
                        p=p,
                        start=list(hof[0]),
                        T0=100.0,
                        Tmin=1e-3,
                        alpha=0.95,
                        iters_per_T=50,
                        seed=random.randrange(1 << 30),
                        hof_size=1,
                        verbose=True)
    return pop_sa, hof_sa, log



# =========================
# 7) GA memético
# =========================
def run_memetic_ga(toolbox,
                   D,
                   pop_size: int = 100,
                   ngen: int = 100,
                   cxpb: float = 0.9,
                   mutpb: float = 0.2,
                   memetic_interval: int = 5,
                   memetic_best_k: int = 5,
                   hof_size: int = 1,
                   seed: int = 0,
                   verbose: bool = True):
    """
    GA memético para tu p-center capacitado.

    Devuelve:
      - pop: población final
      - hof: Hall of Fame (mejores individuos)
      - log: logbook con (gen, min, avg, max)
    """
    rng = random.Random(seed)
    V, nH = D.shape

    log = _init_logbook()
    hof = tools.HallOfFame(hof_size)

    # === Población inicial
    pop = toolbox.population(n=pop_size)

    # Evaluar población inicial
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hof.update(pop)
    fits = [ind.fitness.values[0] for ind in pop]
    _record_log(log, gen=0, fitness_values=fits)

    if verbose:
        print(f"[GA] gen=0 min={min(fits):.6f} avg={sum(fits)/len(fits):.6f} max={max(fits):.6f}")

    # === Bucle evolutivo
    for gen in range(1, ngen + 1):
        # Selección -> como haces en otros algoritmos
        offspring = toolbox.select(pop, len(pop))
        # Clonado (usamos deepcopy, no tools.clone)
        offspring = [deepcopy(ind) for ind in offspring]

        # Crossover
        for i in range(0, len(offspring), 2):
            if i + 1 >= len(offspring):
                break
            if random.random() < cxpb:
                offspring[i], offspring[i+1] = toolbox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        # Mutación
        for ind in offspring:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Evaluar descendencia
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # Elitismo simple: mantén el mejor de pop anterior
        elite = deepcopy(tools.selBest(pop, 1)[0])

        # Reemplazo generacional
        pop = offspring
        # insertar élite
        worst = tools.selWorst(pop, 1)[0]
        pop[pop.index(worst)] = elite

        # === MODO MEMÉTICO: búsqueda local sobre los mejores cada X generaciones
        if memetic_interval > 0 and gen % memetic_interval == 0:
            best_inds = tools.selBest(pop, memetic_best_k)
            for ind in best_inds:
                local_search_1swap_ind(ind, toolbox, nH,
                                       max_iterations=10,
                                       neighbors_per_iteration=5,
                                       rng=rng)

        # Actualizar Hall of Fame
        hof.update(pop)

        # Estadísticas
        fits = [ind.fitness.values[0] for ind in pop]
        _record_log(log, gen=gen, fitness_values=fits)

        if verbose:
            print(f"[GA] gen={gen} min={min(fits):.6f} avg={sum(fits)/len(fits):.6f} max={max(fits):.6f}")

    return pop, hof, log