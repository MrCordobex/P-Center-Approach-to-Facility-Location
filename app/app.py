import streamlit as st
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import random
from streamlit_folium import st_folium

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import utils.utils as ut
import scr.functions as fn
import scr.algorithms as alg
from cache_manager import SolutionCache
from map_generator import generate_map

st.set_page_config(layout="wide", page_title="Andalusia Hospital Optimization")

st.title("Andalusia Hospital Optimization (P-Center)")

# Sidebar
st.sidebar.header("Parameters")
rho = st.sidebar.number_input("Rho (Capacity Buffer)", min_value=0.001, max_value=0.010, value=0.005, step=0.001, format="%.3f")
p = st.sidebar.number_input("P (Number of Hospitals)", min_value=1, max_value=130, value=5, step=1)

# Advanced
with st.sidebar.expander("Advanced Parameters"):
    H_dias = st.number_input("Horizon (Days)", value=30)
    LOS = st.number_input("Length of Stay (Days)", value=5)
    occ = st.number_input("Target Occupancy", value=0.9)

# ACME Hyperparameters (Recalculate)
with st.sidebar.expander("ACME Hyperparameters (Recalculate)"):
    acme_ngen = st.number_input("Generations", value=100, step=10)
    acme_mu = st.number_input("Population Size", value=100, step=10)
    acme_cxpb = st.number_input("Crossover Prob", value=0.9, step=0.05)
    acme_mutpb = st.number_input("Mutation Prob", value=0.3, step=0.05)

col1, col2 = st.sidebar.columns(2)
run_btn = col1.button("Run/Load")
recalc_btn = col2.button("Recalculate")

# --- MAIN LOGIC ---

# Initialize Cache
cache = SolutionCache(os.path.join(current_dir, "solutions.csv"))

# Initialize Session State
if "solution_data" not in st.session_state:
    st.session_state.solution_data = None
if "map_context" not in st.session_state:
    st.session_state.map_context = None

should_run = False
force_recalc = False

if run_btn:
    should_run = True
if recalc_btn:
    should_run = True
    force_recalc = True

if should_run:
    st.info(f"Processing for rho={rho}, p={p}...")

    # 1. Check Cache
    cached_sol = cache.get_solution(rho, p, H_dias, LOS, occ)
    
    new_solution_data = None
    new_map_context = {}
    
    # Load Data
    with st.spinner("Loading and processing data..."):
        base_path = project_root
        processed_dir = os.path.join(base_path, "data", "processed")
        matrix_dir = os.path.join(base_path, "data", "matrix")
        
        ut.capacities_demand(
            ruta_ciudades=os.path.join(processed_dir, "andaluces_2_5k.csv"),
            ruta_hospitales=os.path.join(processed_dir, "Hospitales_Completo.csv"),
            ruta_destino_ciudades=os.path.join(processed_dir, "Ciudades_Con_Demanda.csv"),
            ruta_destino_hospitales=os.path.join(processed_dir, "Hospitales_Con_Capacidad.csv"),
            H=H_dias, LOS=LOS, occ=occ, rho=rho
        )
        
        D, q, C, city_ids, hosp_ids = ut.load_data(
            dist_csv=os.path.join(matrix_dir, "time_ciudad_hospital_min.csv"),
            cities_csv=os.path.join(processed_dir, "Ciudades_Con_Demanda.csv"),
            hospitals_csv=os.path.join(processed_dir, "Hospitales_Con_Capacidad.csv")
        )
        
        cities_df = pd.read_csv(os.path.join(processed_dir, "Ciudades_Con_Demanda.csv"))
        hospitals_df = pd.read_csv(os.path.join(processed_dir, "Hospitales_Con_Capacidad.csv"))
        
        new_map_context = {
            "D": D, "q": q, "C": C,
            "cities_df": cities_df,
            "hospitals_df": hospitals_df
        }

    run_algorithm = False
    
    if force_recalc:
        run_algorithm = True
        st.warning("Recalculating solution with new parameters...")
    elif cached_sol:
        st.success("Solution found in cache!")
        new_solution_data = cached_sol
    else:
        run_algorithm = True
        st.warning("Solution not in cache. Running ACME algorithm... This may take a while.")

    if run_algorithm:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        
        toolbox = fn.build_deap_toolbox(D, q, C, p=p, seed=seed)
        
        # Progress UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        map_placeholder = st.empty()

        def progress_callback(gen, ngen, hof, message=None):
            progress = gen / ngen
            progress_bar.progress(progress)
            
            if message:
                status_text.text(message)
            else:
                status_text.text(f"Generation {gen}/{ngen} - Best Fitness: {hof[0].fitness.values[0]:.2f}")
            
            if gen % 5 == 0:
                best_ind = hof[0]
                hospitals_indices = list(best_ind)
                m = generate_map(D, q, C, hospitals_indices, cities_df, hospitals_df)
                
                key_suffix = "_msg" if message else ""
                with map_placeholder.container():
                    st_folium(m, width=1000, height=600, returned_objects=[], key=f"map_gen_{gen}{key_suffix}")

        with st.spinner("Running ACME..."):
            pop, hof, log = alg.run_acme(
                toolbox,
                p=p,
                D=D, q=q, C=C,
                ngen=acme_ngen,
                mu_pop=acme_mu,
                cxpb=acme_cxpb, 
                mutpb=acme_mutpb,
                hof_size=5,
                k_clusters=5,
                acme_period=5,
                ls_iters=200,
                kmeans_max_iter=10,
                seed=seed,
                verbose=True,
                callback=progress_callback
            )
        
        progress_bar.empty()
        status_text.empty()
        map_placeholder.empty()
        
        best_ind = hof[0]
        hospitals_indices = list(best_ind)
        
        assign, Z, cap_left, penalty_unserved, num_unserved = fn.greedy2_assignment_with_capacities(
            D, q, C, hospitals_indices
        )
        
        fitness = best_ind.fitness.values[0]
        
        should_save = True
        if force_recalc and cached_sol:
            old_fitness = cached_sol["fitness"]
            if fitness < old_fitness:
                st.success(f"New solution is better! (New: {fitness:.2f} < Old: {old_fitness:.2f}). Updating cache.")
            else:
                st.error(f"Optimization failed to improve. (New: {fitness:.2f} >= Old: {old_fitness:.2f}). Keeping old solution.")
                should_save = False
                new_solution_data = cached_sol
        
        if should_save:
            cache.save_solution(rho, p, hospitals_indices, fitness, Z, penalty_unserved, H_dias, LOS, occ)
            new_solution_data = {
                "hospitals_indices": hospitals_indices,
                "fitness": fitness,
                "z_max": Z,
                "unserved_demand": penalty_unserved
            }
            st.success("Optimization finished and saved to cache.")
    
    st.session_state.solution_data = new_solution_data
    st.session_state.map_context = new_map_context
    
    with st.spinner("Generating map..."):
        st.session_state.map_obj = generate_map(
            new_map_context["D"], new_map_context["q"], new_map_context["C"], 
            new_solution_data["hospitals_indices"], 
            new_map_context["cities_df"], new_map_context["hospitals_df"]
        )

# --- VISUALIZATION ---
if st.session_state.solution_data and st.session_state.map_context:
    sol = st.session_state.solution_data
    ctx = st.session_state.map_context
    
    st.subheader("Solution Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Fitness", f"{sol['fitness']:.2f}")
    col2.metric("Max Distance (Z)", f"{sol['z_max']:.2f} min")
    col3.metric("Unserved Demand", f"{sol['unserved_demand']:.2f}")
    
    st.subheader("Map Visualization")
    if "map_obj" in st.session_state:
        st_folium(st.session_state.map_obj, width=1000, height=600, returned_objects=[])
    else:
        st.warning("Map object not found. Please run optimization again.")
        
    st.subheader("Selected Hospitals")
    
    assign, Z, cap_left, penalty_unserved, num_unserved = fn.greedy2_assignment_with_capacities(
        ctx["D"], ctx["q"], ctx["C"], sol["hospitals_indices"]
    )
    
    selected_df = ctx["hospitals_df"].iloc[sol["hospitals_indices"]].copy()
    
    occupancy_list = []
    for idx in sol["hospitals_indices"]:
        capacity = ctx["C"][idx]
        remaining = cap_left.get(idx, capacity)
        used = capacity - remaining
        pct = (used / capacity) * 100 if capacity > 0 else 0
        occupancy_list.append(f"{pct:.1f}%")
        
    selected_df["Occupancy"] = occupancy_list
    
    st.dataframe(selected_df[["nombre", "localidad", "provincia", "C", "Occupancy"]])

elif not run_btn:
    st.info("Adjust parameters and click 'Run Optimization' to start.")
