import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull
from pathlib import Path
from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget

# --- 1. DATA & MODEL SETUP ---
appdir = Path(__file__).parent
data_path = appdir / 'data' / 'train_data.csv'
demo_data_path = appdir / 'data' / 'demo_data.csv'

# Supervised Model
xgb_model = joblib.load(appdir / 'models' / 'xgb_cancer_model.joblib')

# Global Pipeline (Benign vs Malignant)
scaler_global = joblib.load(appdir / 'models' / 'scaler_global.joblib')
pca_global = joblib.load(appdir / 'models' / 'pca_global.joblib')
gmm_global = joblib.load(appdir / 'models' / 'gmm_global.joblib')
global_malig_id = joblib.load(appdir / 'models' / 'global_malig_id.joblib')
df_global_bg = pd.read_csv(appdir / 'data' / 'global_pca_background.csv')

# Malignant Sub-type Pipeline (Unsupervised Second Opinion)
scaler_malig = joblib.load(appdir / 'models' / 'scaler_malig.joblib')
pca_malig = joblib.load(appdir / 'models' / 'pca_malig.joblib')
gmm_malig = joblib.load(appdir / 'models' / 'gmm_malig.joblib')
outlier_threshold = joblib.load(appdir / 'models' / 'outlier_threshold.joblib')
df_malig_bg = pd.read_csv(appdir / 'data' / 'malig_pca_background.csv')

# Load Data
base_df = pd.read_csv(data_path).drop(columns=['id', 'Unnamed: 32'], errors='ignore')
demo_df = pd.read_csv(demo_data_path).drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Predictable patient selection: 1-5 Malignant, 6-10 Benign
patient_choices = {str(i): f"Demo Patient {i+1}" for i in range(len(demo_df))}

# --- 2. UI ARCHITECTURE ---
app_ui = ui.page_navbar(
    ui.head_content(
        ui.tags.style("""
            .card-header { font-weight: bold; }
            .alert { border: none; font-weight: bold; border-radius: 8px; }
            .fs-3 { font-weight: bold; }
        """)
    ),
    ui.nav_panel(
        "Dashboard",
        
        # KPI Cards
        ui.layout_columns(
            ui.card(ui.card_header("Training Cohort", class_="bg-light"), ui.output_text("kpi_total"), class_="text-center fs-3"),
            ui.card(ui.card_header("Malignant Cases", class_="bg-danger text-white"), ui.output_text("kpi_malignant"), class_="text-center fs-3"),
            ui.card(ui.card_header("Benign Cases", class_="bg-primary text-white"), ui.output_text("kpi_benign"), class_="text-center fs-3"),
        ),
        
        ui.br(),
        
        ui.layout_columns(
            # Left Column: Ensemble Logic
            ui.card(
                ui.card_header("Ensemble Consensus", class_="bg-dark text-white"),
                ui.input_select("selected_patient", "Select Patient Profile:", choices=patient_choices),
                ui.hr(),
                
                ui.h6("Layer 1: Supervised (XGBoost)", class_="text-primary"),
                ui.output_text("xgb_result"),
                ui.br(),
                
                ui.h6("Layer 2: Unsupervised (Global GMM)", class_="text-primary"),
                ui.output_text("global_gmm_result"),
                ui.br(),
                
                ui.h6("Layer 3: Morphological Cluster (Sub-type GMM)", class_="text-primary"),
                ui.output_text("subtype_gmm_result"),
                ui.hr(),
                
                ui.h5("Ensemble Decision Status"),
                ui.output_ui("ensemble_status"),
                ui.hr(),
                
                ui.p("Reference Pathology (Ground Truth):", class_="text-muted mb-0"),
                ui.output_text("true_label"),
            ),
            
            # Right Column: Maps
            ui.div(
                ui.card(
                    ui.card_header("Global Topological Mapping (B vs M)", class_="bg-secondary text-white"),
                    output_widget("global_plot", height="400px")
                ),
                ui.card(
                    ui.card_header("Malignant Sub-type Boundaries", class_="bg-secondary text-white"),
                    output_widget("malig_plot", height="400px")
                ),
            ),
            col_widths=(4, 8)
        ),
    ),
    title="Breast Cancer AI Diagnostics",
    id="navbar",
    navbar_options=ui.navbar_options(
        bg="#007bff", 
        theme="dark"
    ),
)

# --- 3. SERVER LOGIC ---
def server(input, output, session):

    @render.text
    def kpi_total(): return f"{len(base_df)} patients"
    
    @render.text
    def kpi_malignant(): return str(len(base_df[base_df["diagnosis"] == "M"]))
    
    @render.text
    def kpi_benign(): return str(len(base_df[base_df["diagnosis"] == "B"]))

    @reactive.calc
    def get_patient():
        return demo_df.iloc[[int(input.selected_patient())]]

    # 1. XGBoost Logic
    @render.text
    def xgb_result():
        X = get_patient().drop(columns=['diagnosis'])
        prob = xgb_model.predict_proba(X)[0][1]
        res = "MALIGNANT" if prob >= 0.35 else "BENIGN"
        return f"{res} ({prob*100:.1f}% risk score)"

    # 2. Global GMM Logic
    @render.text
    def global_gmm_result():
        X = get_patient().drop(columns=['diagnosis'])
        p_pca = pca_global.transform(scaler_global.transform(X))
        pred = gmm_global.predict(p_pca)[0]
        res = "Malignant Territory" if pred == global_malig_id else "Benign Territory"
        return f"Assessment: {res}"

    # 3. Sub-type GMM Logic
    @render.text
    def subtype_gmm_result():
        X = get_patient().drop(columns=['diagnosis'])
        p_pca = pca_malig.transform(scaler_malig.transform(X))
        log_prob = gmm_malig.score_samples(p_pca)[0]
        if log_prob < outlier_threshold:
            return "Assessment: Outlier (Atypical/Non-Malignant)"
        else:
            cluster = gmm_malig.predict(p_pca)[0] + 1
            return f"Assessment: Malignant Sub-type {cluster}"

    # Ensemble logic with Triple-Validation
    @render.ui
    def ensemble_status():
        X = get_patient().drop(columns=['diagnosis'])
        
        # XGBoost Check
        xgb_m = xgb_model.predict_proba(X)[0][1] >= 0.35
        # Global GMM Check
        p_pca_g = pca_global.transform(scaler_global.transform(X))
        global_m = gmm_global.predict(p_pca_g)[0] == global_malig_id
        # Sub-type GMM Check
        p_pca_m = pca_malig.transform(scaler_malig.transform(X))
        subtype_m = gmm_malig.score_samples(p_pca_m)[0] >= outlier_threshold
        
        votes = sum([xgb_m, global_m, subtype_m])
        
        if votes == 3:
            return ui.HTML("<div class='alert alert-success bg-success text-white'>Unanimous Malignant Consensus</div>")
        elif votes == 0:
            return ui.HTML("<div class='alert alert-success bg-success text-white'>Unanimous Benign Consensus</div>")
        elif not xgb_m and votes >= 2:
            return ui.HTML("<div class='alert alert-danger bg-danger text-white'>CRITICAL: High False Negative Risk</div>")
        elif xgb_m and votes < 3:
            return ui.HTML("<div class='alert alert-warning bg-warning text-dark'>CAUTION: Potential False Positive</div>")
        else:
            return ui.HTML("<div class='alert alert-info bg-info text-white'>Manual Pathology Review Required</div>")

    @render.text
    def true_label():
        label = get_patient()['diagnosis'].values[0]
        return "CONFIRMED MALIGNANT" if label == "M" else "CONFIRMED BENIGN"

    # --- PLOT 1: GLOBAL TOPOLOGY (With Boundaries) ---
    @render_widget
    def global_plot():
        X = get_patient().drop(columns=['diagnosis'])
        p_pca = pca_global.transform(scaler_global.transform(X))
        prob = xgb_model.predict_proba(X)[0][1]
        
        # Determine marker color based on XGBoost prediction
        marker_color = '#228B22' if prob < 0.35 else 'black'
        
        # 1. Base Scatter Plot
        fig = px.scatter(df_global_bg, x='pca1', y='pca2', color='Diagnosis',
                         color_discrete_map={'Malignant (M)': '#ef553b', 'Benign (B)': '#636efa'}, 
                         opacity=0.3)
        
        # 2. Add Shaded Convex Hull Boundaries for B and M
        # Map our colors to the Diagnosis labels
        boundary_colors = {'Malignant (M)': '#ef553b', 'Benign (B)': '#636efa'}
        
        for diag_label, color in boundary_colors.items():
            # Filter points for this specific diagnosis
            pts = df_global_bg[df_global_bg['Diagnosis'] == diag_label][['pca1', 'pca2']].values
            
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                # Get the vertices and close the loop
                h_pts = np.append(pts[hull.vertices], [pts[hull.vertices][0]], axis=0)
                
                fig.add_trace(go.Scatter(
                    x=h_pts[:, 0], y=h_pts[:, 1],
                    mode='lines',
                    fill='toself',
                    fillcolor=color,
                    line=dict(color=color, width=1),
                    opacity=0.1, # Keep it subtle so points are still visible
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # 3. Add the Live Patient Marker
        fig.add_trace(go.Scatter(
            x=[p_pca[0, 0]], y=[p_pca[0, 1]],
            mode='markers',
            marker=dict(size=18, color=marker_color, symbol='diamond', 
                        line=dict(color='white', width=2)),
            name="Current Patient"
        ))
        
        fig.update_layout(
            height=400, 
            margin=dict(l=10, r=10, b=10, t=10), 
            showlegend=True,
            paper_bgcolor='white', 
            plot_bgcolor='white'
        )
        return fig

    @render_widget
    def malig_plot():
        X = get_patient().drop(columns=['diagnosis'])
        p_pca = pca_malig.transform(scaler_malig.transform(X))
        log_prob = gmm_malig.score_samples(p_pca)[0]
        
        marker_color = '#228B22' if log_prob < outlier_threshold else 'black'
        
        fig = px.scatter(df_malig_bg, x='pca1', y='pca2', color='Cluster',
                         color_discrete_sequence=px.colors.qualitative.Pastel, opacity=0.5)
        
        # Add Convex Hulls for visual boundaries
        colors = px.colors.qualitative.Pastel
        for i, cluster in enumerate(sorted(df_malig_bg['Cluster'].unique())):
            pts = df_malig_bg[df_malig_bg['Cluster'] == cluster][['pca1', 'pca2']].values
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                h_pts = np.append(pts[hull.vertices], [pts[hull.vertices][0]], axis=0)
                fig.add_trace(go.Scatter(x=h_pts[:,0], y=h_pts[:,1], fill='toself', 
                                         fillcolor=colors[i], opacity=0.2, showlegend=False, hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(x=[p_pca[0,0]], y=[p_pca[0,1]], mode='markers',
                                 marker=dict(size=18, color=marker_color, symbol='diamond', line=dict(color='white', width=2)),
                                 name="Current Patient"))
        
        fig.update_layout(height=400, margin=dict(l=10,r=10,b=10,t=10), showlegend=True,
                          paper_bgcolor='white', plot_bgcolor='white')
        return fig

app = App(app_ui, server)