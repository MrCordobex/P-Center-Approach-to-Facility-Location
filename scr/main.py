import utils.utils as ut
import scr.algorithms as alg
import scr.functions as fn
import pandas as pd
# para ejecutar desde la raíz del proyecto hacemos python -m scr.main
# -----------------------------------------------------------
# Ajuste capacidades y demandas
ut.capacities_demand(
    ruta_ciudades="data/processed/andaluces_2_5k.csv",
    ruta_hospitales="data/processed/Hospitales_Completo.csv",
    ruta_destino_ciudades="data/processed/Ciudades_Con_Demanda.csv",
    ruta_destino_hospitales="data/processed/Hospitales_Con_Capacidad.csv",
    H=30, LOS=5, occ=0.9, rho=0.005
)
# -----------------------------------------------------------
# Parámetros del problema
P = 25  # número de hospitales a abrir (ajústalo)
SEED = 123

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
# -----------------------------------------------------------

pop, hof, log =  alg.run_eaMuCommaLambda(toolbox, ngen=300, mu=100, lambda_=200, cxpb=0.85, mutpb=0.15, hof_size=3, verbose=True)
# -----------------------------------------------------------
#=== Resultados finales ===
# Resultados
best = hof.items[0]
print("\n=== MEJOR SOLUCIÓN ===")
print("Hosp. abiertos (índices):", best)
best_fit = best.fitness.values[0]
print(f"Fitness (Z + penalización): {best_fit:.4f}")

# Decodificar para ver Z y estadísticas finales
assign, Z, cap_left, penalty_unserved, num_unserved = fn.greedy_assignment_with_capacities(D, q, C, best)
print(f"Z (máx. tiempo): {Z:.4f} min")
print(f"Demanda no atendida: {penalty_unserved:.2f} (ciudades sin asignar: {num_unserved})")
# Mapa
_ = ut.create_map_solution(
    D=D, q=q, C=C, open_idx=list(best),
    cities_csv="data/processed/Ciudades_Con_Demanda.csv",
    hospitals_csv="data/processed/Hospitales_Con_Capacidad.csv",
    out_html="runs/solucion_map.html",
    draw_lines=True  # pon True si quieres líneas ciudad→hospital
)
print("Mapa guardado en: runs/solucion_map.html")
# === 1) Estadísticas y gráfica de evolución ===
df_stats = ut.logbook_to_dataframe(log)
df_stats.to_csv("runs/evolucion_fitness.csv", index=False)
print("Estadísticas por generación guardadas en: runs/evolucion_fitness.csv")

# Usa el mismo bigM que diste en toolbox.evaluate (yo usé 1e9 arriba)
ut.plot_evolution_threshold(df_stats, rute="runs/evolucion_fitness.png", threshold=1000.0, replacement_value=300.0)
# === 2) DataFrame de hospitales seleccionados (nombre, localidad) ===
df_sol = ut.solution_dataframe(best, hospitals_csv="data/processed/Hospitales_Con_Capacidad.csv",
                            name_col="nombre", locality_col="localidad")
df_sol.to_csv("runs/solucion.csv", index=False)
print("Solución guardada en: runs/solucion.csv")
print("\n=== Hospitales seleccionados ===")
print(df_sol)
