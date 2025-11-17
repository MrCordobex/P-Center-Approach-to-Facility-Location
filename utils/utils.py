import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import folium
from scr.functions import greedy_assignment_with_capacities

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


import folium
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

def _hex_color_hsv(i: int, n: int) -> str:
    """
    Genera n colores bien espaciados en HSV y devuelve el i-ésimo en HEX.
    Evita colores demasiado claros/oscuro.
    """
    h = (i / max(n, 1)) % 1.0
    s = 0.75
    v = 0.95
    # HSV -> RGB
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

def create_map_solution(
    D: np.ndarray,
    q: np.ndarray,
    C: np.ndarray,
    open_idx: List[int],
    cities_csv: str = "data/processed/Ciudades_Con_Demanda.csv",
    hospitals_csv: str = "data/processed/Hospitales_Con_Capacidad.csv",
    out_html: str = "runs/solucion_map.html",
    draw_lines: bool = False,
    line_opacity: float = 0.45,
    city_radius: int = 4,
    selected_hosp_radius: int = 8,
) -> folium.Map:
    """
    Dibuja:
      - SOLO hospitales seleccionados (open_idx) en negro.
      - Ciudades coloreadas por el hospital seleccionado al que fueron asignadas.
      - Ciudades no asignadas (si hubiera) en gris.
      - Leyenda: 'Nombre del hospital (Localidad)'.
    """
    # Cargar datos
    datos = pd.read_csv(cities_csv).copy()
    hospitales = pd.read_csv(hospitals_csv).copy()

    for df in (datos, hospitales):
        df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
        df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')

    center_lat = float(datos['latitud'].mean())
    center_lon = float(datos['longitud'].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    # Asignación greedy (tu función)
    assign_of_city, Z, cap_left, penalty_unserved, num_unserved = greedy_assignment_with_capacities(
        D, q, C, open_idx
    )

    # Colores por hospital abierto (para pintar ciudades)
    open_idx_sorted = sorted(open_idx)
    color_by_hosp: Dict[int, str] = {
        h: _hex_color_hsv(k, len(open_idx_sorted)) for k, h in enumerate(open_idx_sorted)
    }

    # Carga utilizada por hospital + lista de ciudades por hospital
    V, H = D.shape
    load_used = np.zeros(H, dtype=float)
    cities_per_h = {h: [] for h in open_idx_sorted}
    for i in range(V):
        h = int(assign_of_city[i])
        if h >= 0:
            load_used[h] += float(q[i])
            if h in cities_per_h:
                cities_per_h[h].append(i)

    # === Hospitales seleccionados en negro ===
    for h in open_idx_sorted:
        if h >= len(hospitales):  # por seguridad
            continue
        lat = hospitales.loc[h, 'latitud']
        lon = hospitales.loc[h, 'longitud']
        if pd.isna(lat) or pd.isna(lon):
            continue
        name = hospitales.loc[h, 'nombre'] if 'nombre' in hospitales.columns else f'Hospital {h}'
        loc = hospitales.loc[h, 'localidad'] if 'localidad' in hospitales.columns else ''
        prov = hospitales.loc[h, 'provincia'] if 'provincia' in hospitales.columns else ''
        used = load_used[h]
        cap = float(C[h])
        left = cap_left.get(h, cap - used)

        popup_text = (
            f"<b>{name}</b><br>"
            f"Localidad: {loc}<br>"
            f"Provincia: {prov}<br>"
            f"Asig. ciudades: {len(cities_per_h.get(h, []))}<br>"
            f"Carga usada: {used:,.0f}<br>"
            f"Cap. restante: {left:,.0f}<br>"
            f"Cap. total (H): {cap:,.0f}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=selected_hosp_radius,
            popup=folium.Popup(popup_text, max_width=300),
            color='black',
            fill=True,
            fill_color='black',
            fill_opacity=1.0,
            weight=2
        ).add_to(m)

    # === Ciudades (color según hospital asignado) ===
    for i in range(V):
        if i >= len(datos):
            continue
        lat = datos.loc[i, 'latitud']
        lon = datos.loc[i, 'longitud']
        if pd.isna(lat) or pd.isna(lon):
            continue

        muni = datos.get('municipio', pd.Series([f'Ciudad {i}']*len(datos))).iloc[i]
        prov = datos.get('PROVINCIA', pd.Series(['']*len(datos))).iloc[i]
        popu = datos.get('poblacion', pd.Series([np.nan]*len(datos))).iloc[i]

        h = int(assign_of_city[i])
        if h >= 0:
            color = color_by_hosp.get(h, '#555555')
            popup_text = (
                f"<b>{muni}</b><br>"
                f"Provincia: {prov}<br>"
                f"Población: {int(popu) if pd.notna(popu) else '-'}<br>"
                f"Asignado a: {hospitales.loc[h, 'nombre'] if 'nombre' in hospitales.columns else h}"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=city_radius,
                popup=folium.Popup(popup_text, max_width=260),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75
            ).add_to(m)

            if draw_lines:
                h_lat = hospitales.loc[h, 'latitud']
                h_lon = hospitales.loc[h, 'longitud']
                if pd.notna(h_lat) and pd.notna(h_lon):
                    folium.PolyLine(
                        locations=[(lat, lon), (h_lat, h_lon)],
                        color=color, weight=1.5, opacity=0.45
                    ).add_to(m)
        else:
            # No asignada (gris)
            popup_text = (
                f"<b>{muni}</b><br>"
                f"Provincia: {prov}<br>"
                f"Población: {int(popu) if pd.notna(popu) else '-'}<br>"
                f"<b>NO ASIGNADA</b>"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=city_radius+1,
                popup=folium.Popup(popup_text, max_width=260),
                color='gray',
                fill=True,
                fill_color='gray',
                fill_opacity=0.5
            ).add_to(m)

    # === Leyenda (hospital abierto ↔ color de sus ciudades) ===
    legend_items = []
    for h in open_idx_sorted:
        color = color_by_hosp[h]
        name = hospitales.loc[h, 'nombre'] if 'nombre' in hospitales.columns else f'Hospital {h}'
        loc = hospitales.loc[h, 'localidad'] if 'localidad' in hospitales.columns else ''
        label = f"{name} ({loc})" if (isinstance(loc, str) and len(loc.strip()) > 0) else f"{name}"
        legend_items.append(f'<li><span style="background:{color};"></span>{label}</li>')
    legend_html = f"""
    <div style="
        position: fixed; bottom: 20px; left: 20px; z-index: 9999;
        background: white; padding: 10px 12px; border: 1px solid #ccc; border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 12px; max-height: 300px; overflow:auto;">
      <div style="font-weight:600; margin-bottom:6px;">Hospital ↔ color de sus ciudades</div>
      <ul style="list-style:none; margin:0; padding:0;">
        {''.join(legend_items)}
      </ul>
      <style>
        ul > li {{ margin: 4px 0; display:flex; align-items:center; gap:8px; }}
        ul > li > span {{
            display:inline-block; width:14px; height:14px; border-radius:3px; border:1px solid #888;
        }}
      </style>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(out_html)
    return m
