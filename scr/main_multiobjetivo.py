# main_multiobjetivo.py

import os
os.environ["OMP_NUM_THREADS"] = "1"  # para evitar problemas con MKL / OMP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.utils as ut
import scr.functions as fn
import scr.multiobjetivo as mo


if __name__ == "__main__":
    # ----------------------------------------
    # 1) Parámetros generales
    # ----------------------------------------
    SEED = 42

    # Rango de hospitales abiertos en el multiobjetivo
    P_MIN = 15    # mínimo de hospitales que permites abrir
    P_MAX = 30  # máximo (ajusta según tu problema)

    # Horizonte / capacidad (igual que en tu main)
    H_DIAS = 30
    LOS = 5
    OCC = 0.9
    RHO = 0.005

    # ----------------------------------------
    # 2) Ajuste de capacidades y demandas
    # ----------------------------------------
    ut.capacities_demand(
        ruta_ciudades="data/processed/andaluces_2_5k.csv",
        ruta_hospitales="data/processed/Hospitales_Completo.csv",
        ruta_destino_ciudades="data/processed/Ciudades_Con_Demanda.csv",
        ruta_destino_hospitales="data/processed/Hospitales_Con_Capacidad.csv",
        H=H_DIAS,
        LOS=LOS,
        occ=OCC,
        rho=RHO,
    )

    # ----------------------------------------
    # 3) Cargar datos del problema
    # ----------------------------------------
    D, q, C, city_ids, hosp_ids = ut.load_data(
        dist_csv="data/matrix/time_ciudad_hospital_min.csv",
        cities_csv="data/processed/Ciudades_Con_Demanda.csv",
        hospitals_csv="data/processed/Hospitales_Con_Capacidad.csv",
        city_id_col=None,
        hosp_id_col=None,
    )

    # ----------------------------------------
    # 4) Construir toolbox multiobjetivo
    # ----------------------------------------
    toolbox_mo = mo.build_deap_toolbox_multiobjective(
        D=D,
        q=q,
        C=C,
        p_min=P_MIN,
        p_max=P_MAX,
        seed=SEED,
    )

    # ----------------------------------------
    # 5) Ejecutar NSGA-II multiobjetivo
    # ----------------------------------------
    POP_SIZE = 300
    NGEN = 200
    CXPB = 0.8
    MUTPB = 0.2

    pop, hof, log = mo.run_nsga2_multiobjective(
        toolbox=toolbox_mo,
        pop_size=POP_SIZE,
        ngen=NGEN,
        cxpb=CXPB,
        mutpb=MUTPB,
        seed=SEED,
        verbose=True,
    )

    print("\n=== PARETO FRONT (soluciones no dominadas) ===")
    pareto_data = []
    for i, ind in enumerate(hof, start=1):
        f1, f2 = ind.fitness.values   # f1 = nº hospitales, f2 = Z_penalizado
        p = int(f1)

        open_idx = [j for j, g in enumerate(ind) if g == 1]
        # Recalcular Z y penalización para verlos limpitos
        assign, Z, cap_left, penalty_unserved, num_unserved = fn.greedy2_assignment_with_capacities(
            D, q, C, open_idx
        )

        print(f"{i:3d}) p = {p:3d}, Z_max = {Z:8.4f}, demanda_no_atendida = {penalty_unserved:.2f}, abiertos = {open_idx[:10]}{'...' if len(open_idx) > 10 else ''}")

        pareto_data.append({
            "p": p,
            "Z_max": Z,
            "demanda_no_atendida": penalty_unserved,
            "num_open": p,
            "open_idx": open_idx,
        })

    # Guardar Pareto en CSV
    df_pareto = pd.DataFrame(pareto_data)
    os.makedirs("runs", exist_ok=True)
    df_pareto.to_csv("runs/pareto_front_solutions.csv", index=False)
    print("\nFrente de Pareto guardado en: runs/pareto_front_solutions.csv")

    # ----------------------------------------
    # 6) Dibujar frente de Pareto (p vs Z_max)
    # ----------------------------------------
    # Sólo consideramos soluciones factibles (sin demanda no atendida)
    df_feas = df_pareto[df_pareto["demanda_no_atendida"] <= 1e-6].copy()
    if df_feas.empty:
        print("⚠ No hay soluciones totalmente factibles en el frente (demanda_no_atendida>0). Se dibuja el frente penalizado.")
        xs = df_pareto["p"].values
        ys = df_pareto["Z_max"].values
        title = "Frente de Pareto (incluyendo penalización)"
    else:
        xs = df_feas["p"].values
        ys = df_feas["Z_max"].values
        title = "Frente de Pareto (soluciones factibles)"

    plt.figure()

    # Ordenar por número de hospitales (p) para que la línea tenga sentido
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]

    plt.plot(xs_sorted, ys_sorted, marker="o")  # línea + puntos
    plt.xlabel("Número de hospitales abiertos (p)")
    plt.ylabel("Distancia máxima Z_min (min)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("runs/pareto_front.png", dpi=150)
    plt.close()
    print("Gráfico del frente de Pareto guardado en: runs/pareto_front.png")

