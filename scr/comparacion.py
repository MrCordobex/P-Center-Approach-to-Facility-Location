import utils.utils as ut
import scr.algorithms as alg
import scr.functions as fn
import pandas as pd
import scr.hyperparam_opt as bayes_opt

# para ejecutar desde la raíz del proyecto hacemos python -m scr.main
# -----------------------------------------------------------
# Ajuste capacidades y demandas
ut.capacities_demand(
    ruta_ciudades="data/processed/andaluces_2_5k.csv",
    ruta_hospitales="data/processed/Hospitales_Completo.csv",
    ruta_destino_ciudades="data/processed/Ciudades_Con_Demanda.csv",
    ruta_destino_hospitales="data/processed/Hospitales_Con_Capacidad.csv",
    H=30, LOS=5, occ=0.9, rho=0.001
)
# -----------------------------------------------------------
# Parámetros del problema
P = 10  # número de hospitales a abrir (ajústalo)
SEED = 100

# Cargar datos
D, q, C, city_ids, hosp_ids = ut.load_data(
    dist_csv="data/matrix/time_ciudad_hospital_min.csv",
    cities_csv="data/processed/Ciudades_Con_Demanda.csv",
    hospitals_csv="data/processed/Hospitales_Con_Capacidad.csv",
    city_id_col=None,   # si tienes un campo ID, pon su nombre aquí
    hosp_id_col=None
)

# Construir toolbox
toolbox = fn.build_deap_toolbox(D, q, C, p=P, seed=SEED)

# ==========================
# 3) Elegir hiperparámetros: fijo o búsqueda bayesiana
# ==========================
USE_BAYES_OPT = True   # pon True si quieres usar búsqueda bayesiana

if USE_BAYES_OPT:
    print("\n=== Iniciando búsqueda bayesiana de hiperparámetros (Optuna) ===")
    study = bayes_opt.bayes_opt_ga(
        toolbox=toolbox,
        D=D,
        n_trials=20,   # prueba primero con 10–20, luego puedes subir
        seed=SEED
    )

    print("\n=== Mejores hiperparámetros encontrados ===")
    print(study.best_params)
    print(f"Mejor fitness observado: {study.best_value:.4f}")

    params = study.best_params
    pop_size         = params["pop_size"]
    ngen             = params["ngen"]
    cxpb             = params["cxpb"]
    mutpb            = params["mutpb"]
    memetic_best_k   = params["memetic_best_k"]
    memetic_interval = params["memetic_interval"]

else:
    # Parámetros que tú quieras fijar a mano
    pop_size         = 100
    ngen             = 200
    cxpb             = 0.90
    mutpb            = 0.11
    memetic_interval = 5
    memetic_best_k   = 2

    print("\n=== Ejecutando GA MEMÉTICO con parámetros fijos ===")
    print(f"pop_size={pop_size}, ngen={ngen}, cxpb={cxpb}, mutpb={mutpb}, "
            f"memetic_interval={memetic_interval}, memetic_best_k={memetic_best_k}")

# ==========================
# 4) Ejecutar GA memético final
# ==========================
pop, hof, log = alg.run_memetic_ga(
    toolbox,
    D,
    pop_size=pop_size,
    ngen=ngen,
    cxpb=cxpb,
    mutpb=mutpb,
    memetic_interval=memetic_interval,
    memetic_best_k=memetic_best_k,
)

# ==========================
# 5) Resultados finales
# ==========================
best = hof.items[0]
print("\n=== MEJOR SOLUCIÓN ===")
print("Hosp. abiertos (índices):", best)
best_fit = best.fitness.values[0]
print(f"Fitness (Z + penalización): {best_fit:.4f}")

# Decodificar para ver Z y estadísticas finales
assign, Z, cap_left, penalty_unserved, num_unserved = fn.greedy_assignment_with_capacities(
    D, q, C, best
)
print(f"Z (máx. tiempo): {Z:.4f} min")
print(f"Demanda no atendida: {penalty_unserved:.2f} (ciudades sin asignar: {num_unserved})")

# ==========================
# 6) Mapa de solución
# ==========================
_ = ut.create_map_solution(
    D=D, q=q, C=C, open_idx=list(best),
    cities_csv="data/processed/Ciudades_Con_Demanda.csv",
    hospitals_csv="data/processed/Hospitales_Con_Capacidad.csv",
    out_html="runs/solucion_map.html",
    draw_lines=True  # True si quieres líneas ciudad→hospital
)
print("Mapa guardado en: runs/solucion_map.html")

# ==========================
# 7) Estadísticas y gráfica de evolución
# ==========================
df_stats = ut.logbook_to_dataframe(log)
df_stats.to_csv("runs/evolucion_fitness.csv", index=False)
print("Estadísticas por generación guardadas en: runs/evolucion_fitness.csv")

# Usa el mismo bigM que diste en toolbox.evaluate (por ejemplo 1e9)
ut.plot_evolution_threshold(
    df_stats,
    rute="runs/evolucion_fitness.png",
    threshold=1000.0,
    replacement_value=300.0
)

# ==========================
# 8) DataFrame de hospitales seleccionados
# ==========================
df_sol = ut.solution_dataframe(
    best,
    hospitals_csv="data/processed/Hospitales_Con_Capacidad.csv",
    name_col="nombre",
    locality_col="localidad"
)
df_sol.to_csv("runs/solucion.csv", index=False)
print("Solución guardada en: runs/solucion.csv")
print("\n=== Hospitales seleccionados ===")
print(df_sol)