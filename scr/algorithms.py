import random
import math
from typing import List, Tuple, Dict
from copy import deepcopy
import numpy as np
import pandas as pd
from collections import deque
from deap import base, creator, tools, algorithms
from scr.functions import _mk_ind, _evaluate, _init_logbook, _record_log, _neighbors_swaps


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


