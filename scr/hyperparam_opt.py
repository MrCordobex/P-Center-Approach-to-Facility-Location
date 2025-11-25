import optuna
import random
from copy import deepcopy
from deap import tools

import scr.algorithms as alg
import scr.functions as fn

def bayes_opt_ga(toolbox, D, n_trials=20, seed=42):
    """
    Optimización bayesiana de hiperparámetros DEL ALGORITMO,
    """

    V, H = D.shape
    random.seed(seed)

    def objective(trial):

        # ============================
        # Hiperparámetros a optimizar:
        # ============================
        pop_size = trial.suggest_int("pop_size", 50, 150)
        ngen = trial.suggest_int("ngen", 80, 200)
        cxpb = trial.suggest_float("cxpb", 0.70, 0.95)
        mutpb = trial.suggest_float("mutpb", 0.05, 0.25)
        memetic_best_k = trial.suggest_int("memetic_best_k", 2, 8)
        memetic_interval = trial.suggest_int(
            "memetic_interval",
            max(3, memetic_best_k),
            max(10, memetic_best_k + 4)
        )

        # =====================================
        # EJECUCIÓN DEL ALGORITMO MEMÉTICO
        # =====================================
        pop, hof, log = alg.run_memetic_ga(
            toolbox,
            D,
            pop_size=pop_size,
            ngen=ngen,
            cxpb=cxpb,
            mutpb=mutpb,
            memetic_interval=memetic_interval,
            memetic_best_k=memetic_best_k,
            verbose=False
        )

        best = hof.items[0]
        return best.fitness.values[0]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler
    )
    study.optimize(objective, n_trials=n_trials)

    return study