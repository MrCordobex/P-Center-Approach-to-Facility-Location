# scr/main_experimentos.py

import os
import time
import random
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

import utils.utils as ut
import scr.functions as fn
import scr.algorithms as alg  # aquí deben estar tus run_*


# ==========================
# CONFIGURACIÓN GENERAL
# ==========================

# Casos de estudio: rho -> lista de valores de p
RHO_P_CASES = {
    0.001: [5, 8, 10],
    0.005: [14, 17, 22],
}

# Semillas a usar para cada (algoritmo, rho, p)
SEEDS = [0, 1, 2, 3, 4]

# Parámetros del horizonte (los puedes cambiar si quieres)
H_DIAS = 30
LOS = 5
OCC = 0.9

# Fichero donde se acumulan los resultados
RESULTS_PATH = "runs/experimentos_algoritmos.xlsx"

# Qué algoritmos quieres probar (usa las claves que definimos más abajo)
ALGO_NAMES = [
    "eaSimple",
    "eaMuPlusLambda",
    "eaMuCommaLambda",
    "hill_climbing",
    "simulated_annealing",
    "acme",
    'acme_no_sa'
]

# ==========================
# HIPERPARÁMETROS POR ALGORITMO
# (CAMBIA AQUÍ LO QUE QUIERAS)
# ==========================

EA_SIMPLE_PARAMS = {
    "ngen": 200,
    "mu_pop": 200,
    "cxpb": 0.8,
    "mutpb": 0.2,
    "hof_size": 5,
    "verbose": True,
}

EA_MUPLUSLAMBDA_PARAMS = {
    "ngen": 200,
    "mu": 100,
    "lambda_": 200,
    "cxpb": 0.8,
    "mutpb": 0.2,
    "hof_size": 5,
    "verbose": True,
}

EA_MUCOMMALAMBDA_PARAMS = {
    "ngen": 200,
    "mu": 100,
    "lambda_": 200,
    "cxpb": 0.8,
    "mutpb": 0.2,
    "hof_size": 5,
    "verbose": True,
}

HC_PARAMS = {
    "max_iters": 200,
    "max_neighbors": 300,
    "hof_size": 1,
    "verbose": True,
}

SA_PARAMS = {
    "T0": 10.0,
    "Tmin": 1e-3,
    "alpha": 0.95,
    "iters_per_T": 300,
    "hof_size": 1,
    "verbose": True,
}

ACME_PARAMS = {
    "ngen": 200,
    "mu_pop": 200,
    "cxpb": 0.9,
    "mutpb": 0.3,
    "hof_size": 5,
    "k_clusters": 5,
    "acme_period": 5,
    "ls_iters": 500,
    "kmeans_max_iter": 10,
    "verbose": True,
}

ACME_NO_SA_PARAMS = {
    "ngen": 200,
    "mu_pop": 200,
    "cxpb": 0.9,
    "mutpb": 0.3,
    "hof_size": 5,
    "k_clusters": 5,
    "acme_period": 5,
    "ls_iters": 500,
    "kmeans_max_iter": 10,
    "verbose": True,
}


def load_existing_results(path: str):
    """
    Carga el Excel si existe y devuelve:
      - df_all: DataFrame con todos los resultados previos (o vacío)
      - done: set de claves (algorithm, rho, p, seed) ya ejecutadas
    """
    if os.path.exists(path):
        df_all = pd.read_excel(path)

        # Si el Excel está vacío por lo que sea
        if df_all.empty:
            done = set()
        else:
            # Nos aseguramos de que las columnas existen
            required_cols = {"algorithm", "rho", "p", "seed"}
            if not required_cols.issubset(df_all.columns):
                raise ValueError(
                    f"El Excel {path} existe pero no tiene las columnas mínimas {required_cols}"
                )

            done = set(
                (str(row["algorithm"]), float(row["rho"]), int(row["p"]), int(row["seed"]))
                for _, row in df_all.iterrows()
            )
    else:
        df_all = pd.DataFrame()
        done = set()

    return df_all, done


def run_experiments():
    os.makedirs("runs", exist_ok=True)

    # 1) Cargar resultados existentes (si los hay)
    df_all, done = load_existing_results(RESULTS_PATH)
    print(f"Se han encontrado {len(done)} experimentos previos en el Excel.")

    # 2) Recorrer cada rho y sus valores de p
    for rho, p_list in RHO_P_CASES.items():
        print(f"\n===============================")
        print(f"   Iniciando bloque rho={rho}")
        print(f"===============================")

        # Ajuste de capacidades y demandas para este rho
        ut.capacities_demand(
            ruta_ciudades="data/processed/andaluces_2_5k.csv",
            ruta_hospitales="data/processed/Hospitales_Completo.csv",
            ruta_destino_ciudades="data/processed/Ciudades_Con_Demanda.csv",
            ruta_destino_hospitales="data/processed/Hospitales_Con_Capacidad.csv",
            H=H_DIAS,
            LOS=LOS,
            occ=OCC,
            rho=rho,
        )

        # Cargar datos del problema
        D, q, C, city_ids, hosp_ids = ut.load_data(
            dist_csv="data/matrix/time_ciudad_hospital_min.csv",
            cities_csv="data/processed/Ciudades_Con_Demanda.csv",
            hospitals_csv="data/processed/Hospitales_Con_Capacidad.csv",
            city_id_col=None,
            hosp_id_col=None,
        )
        nH = D.shape[1]

        # 3) Recorremos cada valor de p para este rho
        for p in p_list:
            print(f"\n--- rho={rho}, p={p} ---")

            for seed in SEEDS:
                # Construimos la clave para este experimento
                for algo_name in ALGO_NAMES:
                    key = (algo_name, float(rho), int(p), int(seed))

                    # 3.a) Comprobar si ya existe en el Excel
                    if key in done:
                        print(f"[SKIP] {algo_name}, rho={rho}, p={p}, seed={seed} ya está en el Excel. Se salta.")
                        continue

                    print(f"\n[RUN] {algo_name}, rho={rho}, p={p}, seed={seed}")

                    # Semilla global
                    random.seed(seed)
                    np.random.seed(seed)

                    # Construir toolbox para este (rho, p, seed)
                    toolbox = fn.build_deap_toolbox(D, q, C, p=p, seed=seed)

                    # 4) Ejecutar el algoritmo correspondiente
                    start_time = time.perf_counter()

                    if algo_name == "eaSimple":
                        pop, hof, log = alg.run_eaSimple(
                            toolbox,
                            ngen=EA_SIMPLE_PARAMS["ngen"],
                            mu_pop=EA_SIMPLE_PARAMS["mu_pop"],
                            cxpb=EA_SIMPLE_PARAMS["cxpb"],
                            mutpb=EA_SIMPLE_PARAMS["mutpb"],
                            hof_size=EA_SIMPLE_PARAMS["hof_size"],
                            verbose=EA_SIMPLE_PARAMS["verbose"],
                        )

                    elif algo_name == "eaMuPlusLambda":
                        pop, hof, log = alg.run_eaMuPlusLambda(
                            toolbox,
                            ngen=EA_MUPLUSLAMBDA_PARAMS["ngen"],
                            mu=EA_MUPLUSLAMBDA_PARAMS["mu"],
                            lambda_=EA_MUPLUSLAMBDA_PARAMS["lambda_"],
                            cxpb=EA_MUPLUSLAMBDA_PARAMS["cxpb"],
                            mutpb=EA_MUPLUSLAMBDA_PARAMS["mutpb"],
                            hof_size=EA_MUPLUSLAMBDA_PARAMS["hof_size"],
                            verbose=EA_MUPLUSLAMBDA_PARAMS["verbose"],
                        )

                    elif algo_name == "eaMuCommaLambda":
                        pop, hof, log = alg.run_eaMuCommaLambda(
                            toolbox,
                            ngen=EA_MUCOMMALAMBDA_PARAMS["ngen"],
                            mu=EA_MUCOMMALAMBDA_PARAMS["mu"],
                            lambda_=EA_MUCOMMALAMBDA_PARAMS["lambda_"],
                            cxpb=EA_MUCOMMALAMBDA_PARAMS["cxpb"],
                            mutpb=EA_MUCOMMALAMBDA_PARAMS["mutpb"],
                            hof_size=EA_MUCOMMALAMBDA_PARAMS["hof_size"],
                            verbose=EA_MUCOMMALAMBDA_PARAMS["verbose"],
                        )

                    elif algo_name == "hill_climbing":
                        pop, hof, log = alg.run_hill_climbing(
                            toolbox,
                            nH=nH,
                            p=p,
                            start=None,
                            max_iters=HC_PARAMS["max_iters"],
                            max_neighbors=HC_PARAMS["max_neighbors"],
                            seed=seed,
                            hof_size=HC_PARAMS["hof_size"],
                            verbose=HC_PARAMS["verbose"],
                        )

                    elif algo_name == "simulated_annealing":
                        pop, hof, log = alg.run_simulated_annealing(
                            toolbox,
                            nH=nH,
                            p=p,
                            start=None,
                            T0=SA_PARAMS["T0"],
                            Tmin=SA_PARAMS["Tmin"],
                            alpha=SA_PARAMS["alpha"],
                            iters_per_T=SA_PARAMS["iters_per_T"],
                            seed=seed,
                            hof_size=SA_PARAMS["hof_size"],
                            verbose=SA_PARAMS["verbose"],
                        )

                    elif algo_name == "acme":
                        pop, hof, log = alg.run_acme(
                            toolbox,
                            p=p,
                            D=D,
                            q=q,
                            C=C,
                            ngen=ACME_PARAMS["ngen"],
                            mu_pop=ACME_PARAMS["mu_pop"],
                            cxpb=ACME_PARAMS["cxpb"],
                            mutpb=ACME_PARAMS["mutpb"],
                            hof_size=ACME_PARAMS["hof_size"],
                            k_clusters=ACME_PARAMS["k_clusters"],
                            acme_period=ACME_PARAMS["acme_period"],
                            ls_iters=ACME_PARAMS["ls_iters"],
                            kmeans_max_iter=ACME_PARAMS["kmeans_max_iter"],
                            seed=seed,
                            verbose=ACME_PARAMS["verbose"],
                        )
                    elif algo_name == "acme_no_sa":
                        pop, hof, log = alg.run_acme(
                            toolbox,
                            p=p,
                            D=D,
                            q=q,
                            C=C,
                            ngen=ACME_NO_SA_PARAMS["ngen"],
                            mu_pop=ACME_NO_SA_PARAMS["mu_pop"],
                            cxpb=ACME_NO_SA_PARAMS["cxpb"],
                            mutpb=ACME_NO_SA_PARAMS["mutpb"],
                            hof_size=ACME_NO_SA_PARAMS["hof_size"],
                            k_clusters=ACME_NO_SA_PARAMS["k_clusters"],
                            acme_period=ACME_NO_SA_PARAMS["acme_period"],
                            ls_iters=ACME_NO_SA_PARAMS["ls_iters"],
                            kmeans_max_iter=ACME_NO_SA_PARAMS["kmeans_max_iter"],
                            seed=seed,
                            verbose=ACME_NO_SA_PARAMS["verbose"],
                        )

                    else:
                        raise ValueError(f"Algoritmo desconocido: {algo_name}")

                    elapsed = time.perf_counter() - start_time

                    # Mejor individuo del Hall of Fame
                    best = hof[0]
                    best_fit = float(best.fitness.values[0])

                    # Decodificar para obtener Z y demanda no atendida con tu greedy mejorado
                    assign, Z, cap_left, penalty_unserved, num_unserved = fn.greedy2_assignment_with_capacities(
                        D, q, C, list(best)
                    )

                    print(
                        f"   -> best_fitness = {best_fit:.6f}, Z = {Z:.4f}, "
                        f"demanda_no_atendida = {penalty_unserved:.2f}, "
                        f"no_asignadas = {num_unserved}, "
                        f"t = {elapsed:.2f}s"
                    )

                    # 5) Construir fila de resultados
                    row = {
                        "algorithm": algo_name,
                        "rho": float(rho),
                        "p": int(p),
                        "seed": int(seed),
                        "best_fitness": best_fit,
                        "Z_max": float(Z),
                        "demanda_no_atendida": float(penalty_unserved),
                        "num_ciudades_no_asignadas": int(num_unserved),
                        "time_sec": elapsed,
                        "H_dias": H_DIAS,
                        "LOS": LOS,
                        "occ": OCC,
                    }

                    # 6) Añadir al DataFrame en memoria
                    df_all = pd.concat([df_all, pd.DataFrame([row])], ignore_index=True)

                    # 7) Actualizar el conjunto de experimentos ya hechos
                    done.add(key)

                    # 8) Guardar inmediatamente en Excel
                    df_all.to_excel(RESULTS_PATH, index=False)
                    print(f"   -> Resultado guardado/actualizado en {RESULTS_PATH}")

    print("\nExperimentos terminados.")


if __name__ == "__main__":
    run_experiments()