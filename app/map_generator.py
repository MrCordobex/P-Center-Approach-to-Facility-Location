import folium
import pandas as pd
import numpy as np
import colorsys
from scr.functions import greedy2_assignment_with_capacities

def _hex_color_hsv(i: int, n: int) -> str:
    """
    Genera n colores bien espaciados en HSV y devuelve el i-ésimo en HEX.
    Evita colores demasiado claros/oscuro.
    """
    h = (i / max(n, 1)) % 1.0
    s = 0.75
    v = 0.95
    # HSV -> RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

def generate_map(D, q, C, open_idx, cities_df, hospitals_df):
    # Ensure numeric coordinates
    cities_df['latitud'] = pd.to_numeric(cities_df['latitud'], errors='coerce')
    cities_df['longitud'] = pd.to_numeric(cities_df['longitud'], errors='coerce')
    hospitals_df['latitud'] = pd.to_numeric(hospitals_df['latitud'], errors='coerce')
    hospitals_df['longitud'] = pd.to_numeric(hospitals_df['longitud'], errors='coerce')

    # Center map on Andalusia
    center_lat = cities_df['latitud'].mean()
    center_lon = cities_df['longitud'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7) # Reduced zoom start just in case

    # Calculate bounds to fit all cities
    sw = cities_df[['latitud', 'longitud']].min().values.tolist()
    ne = cities_df[['latitud', 'longitud']].max().values.tolist()
    m.fit_bounds([sw, ne])
    
    # Calculate assignments
    assign_of_city, Z, cap_left, penalty_unserved, num_unserved = greedy2_assignment_with_capacities(
        D, q, C, open_idx
    )

    # Colors for each open hospital
    open_idx_sorted = sorted(open_idx)
    color_by_hosp = {
        h: _hex_color_hsv(k, len(open_idx_sorted)) for k, h in enumerate(open_idx_sorted)
    }

    # 1. Plot Cities and Lines (First, so they are below hospitals)
    for i, row in cities_df.iterrows():
        if pd.isna(row['latitud']) or pd.isna(row['longitud']):
            continue
            
        h = int(assign_of_city[i])
        
        if h >= 0:
            color = color_by_hosp.get(h, '#555555')
            h_row = hospitals_df.iloc[h]
            
            popup_text = (
                f"<b>{row['municipio']}</b><br>"
                f"Demand: {q[i]:.2f}<br>"
                f"Assigned to: {h_row['nombre']}"
            )
            
            # Line to Hospital
            if pd.notna(h_row['latitud']) and pd.notna(h_row['longitud']):
                folium.PolyLine(
                    locations=[(row['latitud'], row['longitud']), (h_row['latitud'], h_row['longitud'])],
                    color=color,
                    weight=1.5,
                    opacity=0.45
                ).add_to(m)

            # City Marker
            folium.CircleMarker(
                location=[row['latitud'], row['longitud']],
                radius=4,
                popup=folium.Popup(popup_text, max_width=260),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75
            ).add_to(m)
            
        else:
            # Unassigned City
            folium.CircleMarker(
                location=[row['latitud'], row['longitud']],
                radius=4,
                popup=f"City: {row['municipio']}<br>Demand: {q[i]:.2f}<br>UNASSIGNED",
                color='gray',
                fill=True,
                fill_color='gray',
                fill_opacity=0.5
            ).add_to(m)

    # 2. Plot Selected Hospitals (Second, so they are on top)
    for h in open_idx_sorted:
        row = hospitals_df.iloc[h]
        if pd.isna(row['latitud']) or pd.isna(row['longitud']):
            continue
            
        popup_text = (
            f"<b>{row['nombre']}</b><br>"
            f"Capacity: {C[h]:.0f}<br>"
            f"Remaining: {cap_left.get(h, 0):.0f}"
        )
        
        # Get the color assigned to this hospital
        h_color = color_by_hosp.get(h, 'black')

        folium.CircleMarker(
            location=[row['latitud'], row['longitud']],
            radius=9,  # Slightly larger
            popup=folium.Popup(popup_text, max_width=300),
            color='black',       # Thick black border
            weight=3,            # Border thickness
            fill=True,
            fill_color=h_color,  # Filled with assigned color
            fill_opacity=1.0
        ).add_to(m)
        
    return m
