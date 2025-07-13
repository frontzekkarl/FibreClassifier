import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import base64
import io
import numpy as np
import pandas as pd
import tifffile
from shapely.geometry import Polygon, Point
import plotly.graph_objs as go
from sklearn.cluster import KMeans

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Fibre Annotation Tool"

# Global state
fibre_polygons = {}
fibre_labels = {}
fibre_intensities = {}
fibre_measurements = {}
current_image = None
file_names = {"image": None, "outline": None}
pixel_area_um2 = 0.5 * 0.5  # adjust if pixel size differs

# --- FUNCTIONS ---

def parse_outline(outline_text):
    lines = outline_text.strip().splitlines()
    fibres = {}
    for i, line in enumerate(lines):
        coords = list(map(int, line.strip().split(',')))
        x = np.array(coords[::2])
        y = np.array(coords[1::2])
        if len(x) >= 3:
            fibres[i + 1] = Polygon(zip(x, y))
    return fibres

color_map = {"positive": "red", "equivocal": "blue", "negative": "green", "unclassified": "gray"}

def generate_overlay():
    fig = go.Figure()
    if current_image is not None:
        fig.add_trace(go.Heatmap(
            z=current_image,
            colorscale='gray',
            showscale=False,
            hoverinfo='skip',
            opacity=1.0
        ))

    cx, cy, text_labels = [], [], []

    for fid, poly in fibre_polygons.items():
        x = list(poly.exterior.xy[0])
        y = list(poly.exterior.xy[1])
        label = fibre_labels.get(fid, "unclassified")
        colour = color_map.get(label, "gray")

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            fill='toself',
            fillcolor=colour,
            opacity=0.3,
            line=dict(color=colour, width=2),
            name=f"Fibre {fid}",
            customdata=[fid] * len(x),
            hoverinfo='text',
            text=f"Fibre {fid} ({label})",
            showlegend=False
        ))

        c = poly.centroid
        cx.append(c.x)
        cy.append(c.y)
        text_labels.append(str(fid))

    fig.add_trace(go.Scatter(
        x=cx, y=cy,
        mode="text",
        text=text_labels,
        textposition="middle center",
        textfont=dict(size=10, color="white"),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title="Fibre Overlay",
        height=400,
        margin=dict(t=40, b=0, l=0, r=0),
        yaxis=dict(scaleanchor="x", autorange="reversed", visible=False),
        xaxis=dict(visible=False),
        dragmode="pan"
    )

    return fig

def generate_raw_image():
    fig = go.Figure()
    if current_image is not None:
        fig.add_trace(go.Heatmap(
            z=current_image,
            colorscale="gray",
            showscale=False,
            hoverinfo="skip"
        ))

    fig.update_layout(
        title="Raw TIFF Image",
        height=400,
        margin=dict(t=20, b=0, l=0, r=0),
        yaxis=dict(scaleanchor="x", autorange="reversed", visible=False),
        xaxis=dict(visible=False),
        dragmode=False
    )
    return fig

# --- LAYOUT ---

app.layout = dbc.Container([
    html.H3("Fibre Classification Dashboard"),
    dcc.Store(id="selected-fibre-id", storage_type="memory"),
    dbc.Row([
        dbc.Col([
            dcc.Upload(id="upload-image", children=html.Div("Upload 8-bit TIFF"),
                       accept=".tif,.tiff", style={"border": "1px solid #aaa", "padding": "5px"}),
            dcc.Upload(id="upload-outline", children=html.Div("Upload Fibre Outline (.txt)"),
                       accept=".txt", style={"border": "1px solid #aaa", "padding": "5px"}),
            html.Br(),
            html.Div(id="file-status", style={"whiteSpace": "pre-line", "fontSize": 12}),
            html.Hr(),
            html.Label("Assign Label to Clicked Fibre (or press E, P, N):"),
            dcc.Dropdown(id="label-dropdown",
                         options=[{"label": lbl, "value": lbl} for lbl in ["positive", "equivocal", "negative"]],
                         value="positive", clearable=False),
            html.Br(),
            html.Button("Download CSV", id="download-button", n_clicks=0),
            dcc.Download(id="download-data"),
        ], width=3),

        dbc.Col([
            html.Div([
                html.H6("Raw TIFF Image"),
                dcc.Graph(id="pure-image", config={"displayModeBar": False}),
                html.Hr(),
                html.H6("Fibre Overlay"),
                dcc.Graph(id="fibre-graph", config={"displayModeBar": False})
            ])
        ], width=9)
    ])
], fluid=True)

# --- CALLBACKS ---

@app.callback(
    Output("file-status", "children"),
    Output("pure-image", "figure"),
    Output("fibre-graph", "figure"),
    Input("upload-image", "contents"),
    Input("upload-outline", "contents"),
    State("upload-image", "filename"),
    State("upload-outline", "filename")
)
def handle_file_upload(image_content, outline_content, image_name, outline_name):
    global fibre_polygons, fibre_labels, current_image, file_names, fibre_intensities, fibre_measurements

    if not image_content or not outline_content:
        return "Waiting for both files...", go.Figure(), go.Figure()

    image_data = base64.b64decode(image_content.split(",")[1])
    current_image = tifffile.imread(io.BytesIO(image_data))
    if current_image.ndim > 2:
        current_image = current_image[0]

    outline_data = base64.b64decode(outline_content.split(",")[1]).decode("utf-8")
    fibre_polygons = parse_outline(outline_data)
    file_names = {"image": image_name, "outline": outline_name}

    fibre_intensities = {}
    fibre_measurements = {}
    for fid, poly in fibre_polygons.items():
        mask = np.zeros_like(current_image, dtype=bool)
        rr, cc = zip(*list(poly.exterior.coords))
        rr = np.clip(np.array(rr, dtype=int), 0, current_image.shape[1] - 1)
        cc = np.clip(np.array(cc, dtype=int), 0, current_image.shape[0] - 1)
        mask[cc, rr] = True
        pixels = current_image[mask]

        mean_val = pixels.mean() if pixels.size else 0
        std_val = pixels.std() if pixels.size else 0
        area_px = pixels.size
        perimeter_px = poly.length
        convex_area_px = poly.convex_hull.area
        area_um2 = area_px * pixel_area_um2
        convex_area_um2 = convex_area_px * pixel_area_um2
        convexity_ratio = area_um2 / convex_area_um2 if convex_area_um2 > 0 else np.nan
        roundness = (4 * np.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else np.nan
        centroid = poly.centroid

        fibre_intensities[fid] = mean_val
        fibre_measurements[fid] = {
            "mean_intensity": round(mean_val, 2),
            "std_intensity": round(std_val, 2),
            "area_um2": round(area_um2, 2),
            "n_pixels": int(area_px),
            "convexity_ratio": round(convexity_ratio, 4),
            "roundness": round(roundness, 4),
            "centroid_x": round(centroid.x, 2),
            "centroid_y": round(centroid.y, 2)
        }

    X = np.array(list(fibre_intensities.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(X)
    labels = kmeans.labels_
    cluster_means = pd.Series(X.flatten()).groupby(labels).mean()
    sorted_clusters = cluster_means.sort_values().index.tolist()
    cluster_to_class = {
        sorted_clusters[0]: "positive",
        sorted_clusters[1]: "equivocal",
        sorted_clusters[2]: "negative"
    }
    fibre_labels = {fid: cluster_to_class[cluster] for fid, cluster in zip(fibre_polygons.keys(), labels)}

    return f"Loaded: {image_name}\nOutline: {outline_name}\n{len(fibre_polygons)} fibres detected.", generate_raw_image(), generate_overlay()

@app.callback(
    Output("fibre-graph", "figure", allow_duplicate=True),
    Output("selected-fibre-id", "data"),
    Output("label-dropdown", "value"),
    Input("fibre-graph", "clickData"),
    Input("label-dropdown", "value"),
    State("selected-fibre-id", "data"),
    prevent_initial_call=True
)
def update_classification(clickData, selected_label, current_fid):
    global fibre_labels
    triggered = ctx.triggered_id

    if triggered == "fibre-graph" and clickData and "points" in clickData:
        pt = clickData["points"][0]
        fid = pt.get("customdata")
        if isinstance(fid, list):
            fid = fid[0]
        fid = int(fid)
        return generate_overlay(), fid, fibre_labels.get(fid, "unclassified")

    elif triggered == "label-dropdown" and current_fid is not None:
        fibre_labels[int(current_fid)] = selected_label
        return generate_overlay(), current_fid, selected_label

    raise dash.exceptions.PreventUpdate

@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_csv(n_clicks):
    if not fibre_labels:
        raise dash.exceptions.PreventUpdate

    df_rows = []
    for fid, label in fibre_labels.items():
        row = {"fibre_id": fid, "label": label}
        row.update(fibre_measurements.get(fid, {}))
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    return dcc.send_data_frame(df.to_csv, "fibre_classifications.csv", index=False)

if __name__ == "__main__":
    app.run(debug=True)
