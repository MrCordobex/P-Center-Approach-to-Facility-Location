import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import folium

def logbook_to_dataframe(log) -> pd.DataFrame:
    """Convierte el logbook de DEAP a DataFrame con columnas gen, min, avg, max, std (si existen)."""
    cols = ["gen"]
    data = {"gen": log.select("gen")}
    for stat in ["min", "avg", "max", "std"]:
        try:
            data[stat] = log.select(stat)
            cols.append(stat)
        except Exception:
            pass
    df = pd.DataFrame(data, columns=cols)
    return df

def plot_evolution_threshold(df_stats: pd.DataFrame,
                             rute: str = "evolucion_fitness.png",
                             threshold: float = 1000.0,
                             replacement_value: float = 300.0):
    """
    Grafica min/avg/max por generación. Cualquier valor > threshold
    se reemplaza por replacement_value únicamente para la visualización.
    """
    if df_stats.empty:
        print("No hay estadísticas para graficar.")
        return

    gens  = df_stats["gen"].to_numpy()
    cols = [c for c in ["min", "avg", "max"] if c in df_stats.columns]

    plt.figure()
    for c in cols:
        arr = df_stats[c].to_numpy(dtype=float).copy()
        mask = np.isfinite(arr) & (arr > threshold)
        arr[mask] = replacement_value
        plt.plot(gens, arr, label=c)

    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title(f"Evolución del fitness (>{threshold:g} → {replacement_value:g})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(rute, dpi=150)
    plt.close()
    print(f"Gráfica de evolución guardada en: {rute}")

def solution_dataframe(best_indices: List[int], hospitals_csv: str = "Hospitales_Con_Capacidad.csv",
                       name_col: str = "nombre", locality_col: str = "localidad") -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas [nombre, localidad] de los hospitales seleccionados (por índice).
    Asume que el orden de las filas del CSV coincide con el índice de los genes.
    """
    dfh = pd.read_csv(hospitals_csv)
    needed_cols = [name_col, locality_col]
    for col in needed_cols:
        if col not in dfh.columns:
            raise ValueError(f"En '{hospitals_csv}' no se encuentra la columna '{col}'.")
    out = dfh.loc[best_indices, needed_cols].reset_index(drop=True)
    return out

def load_data(
    dist_csv: str = "time_ciudad_hospital_min.csv",
    cities_csv: str = "Ciudades_Con_Demanda.csv",
    hospitals_csv: str = "Hospitales_Con_Capacidad.csv",
    city_id_col: str = None,         # si tienes IDs explícitos, indícalos; si no, se usa el orden del CSV
    hosp_id_col: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, List]:
    """
    Carga:
      - D: matriz |V| x |H| con distancias (min).
      - q: vector |V| con demandas.
      - C: vector |H| con capacidades.
    Devuelve además listas con IDs (si existen) para trazabilidad.
    """
    # Matriz de distancias: filas=ciudades, columnas=hospitales
    D = pd.read_csv(dist_csv, header=None).values.astype(float)  # ajusta header=None según tus archivos

    df_cities = pd.read_csv(cities_csv)
    df_hosp = pd.read_csv(hospitals_csv)

    if 'q' not in df_cities.columns:
        raise ValueError("El CSV de municipios debe tener columna 'q' con las demandas.")
    if 'C' not in df_hosp.columns:
        raise ValueError("El CSV de hospitales debe tener columna 'C' con las capacidades.")

    q = df_cities['q'].values.astype(float)
    C = df_hosp['C'].values.astype(float)

    # IDs opcionales (para auditoría)
    city_ids = df_cities[city_id_col].tolist() if city_id_col and city_id_col in df_cities.columns else list(range(len(q)))
    hosp_ids = df_hosp[hosp_id_col].tolist() if hosp_id_col and hosp_id_col in df_hosp.columns else list(range(len(C)))

    # Chequeos básicos
    nV, nH = D.shape
    if nV != len(q):
        raise ValueError(f"Las filas de la matriz de distancias ({nV}) deben coincidir con |V|=len(q) ({len(q)}).")
    if nH != len(C):
        raise ValueError(f"Las columnas de la matriz de distancias ({nH}) deben coincidir con |H|=len(C) ({len(C)}).")

    return D, q, C, city_ids, hosp_ids



def create_map(datos: pd.DataFrame, hospitales: pd.DataFrame, out_html: str):
    # Asegurar que latitud y longitud son numéricas
    datos['latitud'] = pd.to_numeric(datos['latitud'], errors='coerce')
    datos['longitud'] = pd.to_numeric(datos['longitud'], errors='coerce')

    hospitales['latitud'] = pd.to_numeric(hospitales['latitud'], errors='coerce')
    hospitales['longitud'] = pd.to_numeric(hospitales['longitud'], errors='coerce')

    # Centro del mapa = media de coordenadas de municipios (o conjunta)
    center_lat = datos['latitud'].mean()
    center_lon = datos['longitud'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    # --- Municipios en azul ---
    for _, row in datos.iterrows():
        if pd.isna(row['latitud']) or pd.isna(row['longitud']):
            continue
        popup_text = (
            f"<b>{row['municipio']}</b><br>"
            f"Provincia: {row['PROVINCIA']}<br>"
            f"Población: {int(row['poblacion'])}<br>"
            f"Lat, Lon: {row['latitud']:.5f}, {row['longitud']:.5f}"
        )
        folium.CircleMarker(
            location=[row['latitud'], row['longitud']],
            radius=4,
            popup=folium.Popup(popup_text, max_width=250),
            fill=True,
            color='blue',
            fill_color='blue',
            fill_opacity=0.6
        ).add_to(m)

    # --- Hospitales en rojo ---
    for _, row in hospitales.iterrows():
        if pd.isna(row['latitud']) or pd.isna(row['longitud']):
            continue
        popup_text = (
            f"<b>{row['nombre']}</b><br>"
            f"Localidad: {row['localidad']}<br>"
            f"Provincia: {row['provincia']}<br>"
            f"Lat, Lon: {row['latitud']:.5f}, {row['longitud']:.5f}"
        )
        folium.CircleMarker(
            location=[row['latitud'], row['longitud']],
            radius=6,
            popup=folium.Popup(popup_text, max_width=250),
            fill=True,
            color='red',
            fill_color='red',
            fill_opacity=0.9
        ).add_to(m)

    m.save(out_html)
    return m


def capacities_demand(ruta_ciudades: str = "../data/processed/andaluces_2_5k.csv",
                      ruta_hospitales: str = "../data/processed/Hospitales_Completo.csv",
                      ruta_destino_ciudades: str = "../data/processed/Ciudades_Con_Demanda.csv",
                      ruta_destino_hospitales: str = "../data/processed/Hospitales_Con_Capacidad.csv",
                      H: int =30, LOS: int =5, occ: float =0.85, rho: float =0.01):
        # Parámetros del horizonte
    H   = H       # días
    LOS = LOS        # días de estancia media por paciente
    occ = occ     # ocupación objetivo (85%)
    rho = rho     # fracción de la población que requerirá ingreso/atención en H

    # Carga
    ciudades   = pd.read_csv(ruta_ciudades)          # debe tener 'poblacion'
    hospitales = pd.read_csv(ruta_hospitales)     # debe tener 'capacidad'
    # Demanda: personas que requerirán la atención en H (mismo tipo que el recurso)
    ciudades["q"] = ciudades["poblacion"] * rho

    # Capacidad: personas que pueden ser atendidas en H con rotación por LOS y ocupación
    hospitales["C"] = hospitales["capacidad"] * (H / LOS) * occ

    # Chequeo de factibilidad global
    Q = ciudades["q"].sum()
    Ctot = hospitales["C"].sum()
    print(f"Demanda total esperada (personas en {H} días): {Q:.1f}")
    print(f"Capacidad total disponible (personas en {H} días): {Ctot:.1f}  ->  Q/C = {Q/Ctot:.2f}")

    # (Opcional) Reescalar demanda si no cabe
    if Q > Ctot:
        factor = Ctot / Q
        ciudades["q"] *= factor
        print(f"Demanda reescalada por factor {factor:.3f} para cumplir factibilidad global.")

    # Guardar para tu GA
    ciudades.to_csv(ruta_destino_ciudades, index=False)
    hospitales.to_csv(ruta_destino_hospitales, index=False)