# fichier: capteurs_cartes.py
import pandas as pd
import folium
from geopy.distance import geodesic
import os

# ── CONFIGURATION DES FICHIERS ───────────────────────────────
FILE_PATH = "data/PEMS07/PeMSD7_M_Station_Info.csv"  
OUTPUT_FILE = "results/capteurs_carte_pems07.html"

# ─────────────────────────────────────────────
# CONFIGURATION COULEURS & PRIORITÉ
# ─────────────────────────────────────────────

FWY_COLORS = {}
ROUTE_PALETTE = [
    "#E63946", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#F44336", "#8BC34A", "#FF5722", "#3F51B5",
    "#009688", "#FFC107", "#607D8B", "#E91E63", "#795548",
]

def get_fwy_color(fwy):
    if fwy not in FWY_COLORS:
        idx = len(FWY_COLORS) % len(ROUTE_PALETTE)
        FWY_COLORS[fwy] = ROUTE_PALETTE[idx]
    return FWY_COLORS[fwy]

def get_priority(row, data):
    fwy_counts = data['Fwy'].value_counts()
    top_fwys = fwy_counts.head(3).index.tolist()
    # Gestion des directions (parfois NB/SB ou N/S dans les fichiers PeMS)
    if row['Fwy'] in top_fwys and row['Dir'] in ['N', 'S', 'NB', 'SB']:
        return "HIGH"
    elif row['Dir'] in ['E', 'W', 'EB', 'WB'] or row['Fwy'] in fwy_counts.head(6).index:
        return "MEDIUM"
    else:
        return "LOW"

PRIORITY_CONFIG = {
    "HIGH":   {"color": "#FF3333", "icon": "star",       "size": 12, "opacity": 1.0},
    "MEDIUM": {"color": "#FF9900", "icon": "info-sign",  "size": 8,  "opacity": 0.85},
    "LOW":    {"color": "#33BB33", "icon": "map-marker", "size": 6,  "opacity": 0.7},
}

# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Charger depuis le fichier CSV ─────────────────────
    if not os.path.exists(FILE_PATH):
        print(f"❌ Erreur : Le fichier '{FILE_PATH}' est introuvable.")
        exit()

    # Note: On utilise index_col=0 car votre fichier commence par une colonne d'index vide
    data = pd.read_csv(FILE_PATH, index_col=0)
    print(f"✅ {len(data)} capteurs chargés depuis le fichier local")

    # ── 2. Créer la carte ─────────────────────────────────────
    lat_mean = data['Latitude'].mean()
    lon_mean = data['Longitude'].mean()

    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=11, tiles=None)

    folium.TileLayer("CartoDB.DarkMatter", name="🌑 Dark (défaut)", show=True).add_to(m)
    folium.TileLayer("CartoDB.Positron",   name="⬜ Clair").add_to(m)
    folium.TileLayer("OpenStreetMap",      name="🗺️ OpenStreetMap").add_to(m)

    # ── 3. Groupes par priorité ───────────────────────────────
    priority_groups = {
        "HIGH":   folium.FeatureGroup(name="🔴 Capteurs Priorité HAUTE",   show=True),
        "MEDIUM": folium.FeatureGroup(name="🟠 Capteurs Priorité MOYENNE", show=True),
        "LOW":    folium.FeatureGroup(name="🟢 Capteurs Priorité BASSE",   show=True),
    }
    link_groups = {}

    # ── 4. Ajouter les capteurs ───────────────────────────────
    data['Priority'] = data.apply(lambda row: get_priority(row, data), axis=1)

    for idx, row in data.iterrows():
        priority = row['Priority']
        cfg = PRIORITY_CONFIG[priority]
        fwy_color = get_fwy_color(row['Fwy'])

        popup_html = f"""
        <div style="font-family:'Segoe UI',sans-serif; min-width:200px;">
          <h4 style="margin:0 0 8px; color:{fwy_color}; border-bottom:2px solid {fwy_color}; padding-bottom:4px;">
            📍 Capteur #{int(row['ID'])}
          </h4>
          <table style="width:100%; font-size:13px; border-collapse:collapse;">
            <tr><td style="padding:3px 6px; color:#888;">Source</td>
                <td style="padding:3px 6px; font-weight:bold; color:#4CAF50;">📄 CSV Local</td></tr>
            <tr style="background:#f5f5f5;">
                <td style="padding:3px 6px; color:#888;">Autoroute</td>
                <td style="padding:3px 6px; font-weight:bold; color:{fwy_color};">Fwy {row['Fwy']}</td></tr>
            <tr><td style="padding:3px 6px; color:#888;">Direction</td>
                <td style="padding:3px 6px; font-weight:bold;">{row['Dir']}</td></tr>
            <tr style="background:#f5f5f5;">
                <td style="padding:3px 6px; color:#888;">Position</td>
                <td style="padding:3px 6px;">{row['Latitude']:.4f}, {row['Longitude']:.4f}</td></tr>
            <tr><td style="padding:3px 6px; color:#888;">Priorité</td>
                <td style="padding:3px 6px;">
                  <span style="background:{cfg['color']}; color:white; padding:2px 8px;
                         border-radius:12px; font-size:11px; font-weight:bold;">
                    {priority}
                  </span>
                </td></tr>
          </table>
        </div>
        """

        # Marqueur Icone
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"Fwy {row['Fwy']} | {priority}",
            icon=folium.Icon(
                color="red" if priority == "HIGH" else "orange" if priority == "MEDIUM" else "green",
                icon=cfg['icon']
            )
        ).add_to(priority_groups[priority])

        # Cercle de style pour l'esthétique
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=cfg['size'], color=fwy_color,
            fill=True, fill_color=fwy_color,
            fill_opacity=0.4, weight=2, opacity=cfg['opacity']
        ).add_to(priority_groups[priority])

    # ── 5. Liaisons entre capteurs (Seuil 3km) ────────────────────
    SEUIL_KM = 3.0
    fwy_list = data['Fwy'].unique()

    for fwy in fwy_list:
        link_groups[fwy] = folium.FeatureGroup(name=f"🔗 Liaisons Fwy {fwy}", show=True)

    # Optimisation simple : on ne compare que les capteurs d'une même autoroute
    for fwy in fwy_list:
        subset = data[data['Fwy'] == fwy].copy()
        coords = subset[['Latitude', 'Longitude']].values
        ids = subset.index.tolist()
        
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                d = geodesic(coords[i], coords[j]).km
                if d <= SEUIL_KM:
                    folium.PolyLine(
                        locations=[coords[i], coords[j]],
                        color=get_fwy_color(fwy),
                        weight=2, opacity=0.5,
                        tooltip=f"Liaison Fwy {fwy} : {d:.2f} km"
                    ).add_to(link_groups[fwy])

    # ── 6. Légende ────────────────────────────────────────────
    legend_fwy_items = "".join([
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
        f'<span style="display:inline-block;width:18px;height:4px;background:{color};border-radius:2px;"></span>'
        f'<span style="font-size:12px;">Fwy {fwy}</span></div>'
        for fwy, color in sorted(FWY_COLORS.items())
    ])

    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
        background:rgba(20,20,30,0.92); color:#eee; border-radius:12px;
        padding:16px 20px; box-shadow:0 4px 24px rgba(0,0,0,0.5);
        font-family:'Segoe UI',sans-serif; min-width:200px;
        border:1px solid rgba(255,255,255,0.08);">
      <div style="font-size:14px; font-weight:700; margin-bottom:10px; color:#fff;">📡 Légende PeMS07</div>
      <div style="font-size:10px; color:#4CAF50; margin-bottom:8px;">📄 Source : CSV Local</div>
      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
        <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#FF3333;"></span>
        <span style="font-size:12px;">HAUTE Priorité</span>
      </div>
      <hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:10px 0;">
      <div style="font-size:11px; text-transform:uppercase; color:#aaa; margin-bottom:6px;">Autoroutes</div>
      {legend_fwy_items}
      <hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:10px 0;">
      <div style="font-size:10px; color:#666;">Capteurs total : {len(data)}</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── 7. Finalisation ───────────────────────────────────────
    for grp in priority_groups.values(): grp.add_to(m)
    for grp in link_groups.values(): grp.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    m.save(OUTPUT_FILE)
    print(f"✅ Carte générée avec succès : {OUTPUT_FILE}")