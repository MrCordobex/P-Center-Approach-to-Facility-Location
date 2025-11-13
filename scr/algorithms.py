import random
import math
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from deap import base, creator, tools, algorithms


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


