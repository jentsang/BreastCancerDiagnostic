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

# Models & Pipelines
xgb_model = joblib.load(appdir / 'models' / 'xgb_cancer_model.joblib')
scaler_global = joblib.load(appdir / 'models' / 'scaler_global.joblib')
pca_global = joblib.load(appdir / 'models' / 'pca_global.joblib')
gmm_global = joblib.load(appdir / 'models' / 'gmm_global.joblib')
global_malig_id = joblib.load(appdir / 'models' / 'global_malig_id.joblib')
df_global_bg = pd.read_csv(appdir / 'data' / 'global_pca_background.csv')

scaler_malig = joblib.load(appdir / 'models' / 'scaler_malig.joblib')
pca_malig = joblib.load(appdir / 'models' / 'pca_malig.joblib')
gmm_malig = joblib.load(appdir / 'models' / 'gmm_malig.joblib')
outlier_threshold = joblib.load(appdir / 'models' / 'outlier_threshold.joblib')
df_malig_bg = pd.read_csv(appdir / 'data' / 'malig_pca_background.csv')

demo_df = pd.read_csv(demo_data_path).drop(columns=['id', 'Unnamed: 32'], errors='ignore')
patient_choices = {str(i): f"Demo Patient {i+1}" for i in range(len(demo_df))}

# Tableau 10 Color Palette
TABLEAU10 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# --- 2. UI ARCHITECTURE ---
app_ui = ui.page_navbar(
    ui.head_content(
        ui.tags.style("""
            .card-header { font-weight: bold; }
            .alert { border: none; font-weight: bold; border-radius: 8px; }
            .fs-2 { font-weight: bold; font-size: 1.8rem !important; }
            .label-sm { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; }
            .bslib-value-box { 
                height: 160px !important; 
                border-radius: 12px; 
            }
            .bg-orange { background-color: #ff7f0e !important; color: white !important; }
            .bg-pink { background-color: #e377c2 !important; color: white !important; }
        """)
    ),
    ui.nav_panel(
        "Dashboard",
        
        # TOP ROW: Standardized Value Boxes
        ui.layout_columns(
            ui.output_ui("box_xgb"),
            ui.output_ui("box_global"),
            ui.output_ui("box_subtype"),
        ),
        
        ui.br(),
        
        ui.layout_columns(
            ui.card(
                ui.card_header("Clinical Decision Support", class_="bg-primary text-white"),
                ui.input_select("selected_patient", "Select Patient Profile:", choices=patient_choices),
                ui.hr(),
                ui.h5("Ensemble Decision Status"),
                ui.output_ui("ensemble_status"),
                ui.hr(),
                ui.div(
                    ui.p("Reference Ground Truth:", class_="label-sm mb-0"),
                    ui.output_text("true_label"),
                    class_="p-2 bg-light rounded"
                ),
                ui.hr(),
                ui.h5("Key Diagnostic Drivers"),
                ui.output_ui("feature_drivers"),
            ),
            ui.div(
                ui.card(
                    ui.card_header("Cytology Risk Map (Benign vs Malignant) - Cluster Model 1", class_="bg-primary text-white"),
                    output_widget("global_plot", height="400px")
                ),
                ui.card(
                    ui.card_header("Malignant Biopsy Risk Stratification - Cluster Model 1", class_="bg-primary text-white"),
                    output_widget("malig_plot", height="400px")
                ),
            ),
            col_widths=(4, 8)
        ),
    ),
    title="Breast Cancer AI Diagnostics",
    id="navbar",
    navbar_options=ui.navbar_options(bg="#1f77b4", theme="dark"),
)

# --- 3. SERVER LOGIC ---
def server(input, output, session):

    @reactive.calc
    def get_patient():
        return demo_df.iloc[[int(input.selected_patient())]]

    # --- DYNAMIC VALUE BOX 1: XGBOOST (Confidence Focus) ---
    @render.ui
    def box_xgb():
        X = get_patient().drop(columns=['diagnosis'])
        probs = xgb_model.predict_proba(X)[0]
        max_prob = np.max(probs)
        prediction = "MALIGNANCY" if np.argmax(probs) == 1 else "BENIGN"
        
        theme = "success"
        if prediction == "MALIGNANCY":
            theme = "warning" if max_prob < 0.70 else "pink"
            
        return ui.value_box(
            "Prediction Model",
            f"{max_prob*100:.1f}%",
            f"SUGGESTIVE OF {prediction}",
            theme=theme,
            class_="fs-2"
        )

    # --- DYNAMIC VALUE BOX 2: GLOBAL GMM (Confidence Focus) ---
    @render.ui
    def box_global():
        X = get_patient().drop(columns=['diagnosis'])
        p_pca = pca_global.transform(scaler_global.transform(X))
        probs = gmm_global.predict_proba(p_pca)[0]
        max_prob = np.max(probs)
        pred = gmm_global.predict(p_pca)[0]
        
        is_malignant = (pred == global_malig_id)
        theme = "pink" if is_malignant else "success"
        label = "MALIGNANT ZONE" if is_malignant else "BENIGN ZONE"
        
        return ui.value_box(
            "Cluster Model 1",
            f"{max_prob*100:.1f}%",
            label,
            theme=theme,
            class_="fs-2"
        )

    # --- DYNAMIC VALUE BOX 3: SUB-TYPE GMM (Confidence Focus) ---
    @render.ui
    def box_subtype():
        X = get_patient().drop(columns=['diagnosis'])
        p_pca = pca_malig.transform(scaler_malig.transform(X))
        
        log_prob = gmm_malig.score_samples(p_pca)[0]
        is_atypical = (log_prob < outlier_threshold)
        
        probs = gmm_malig.predict_proba(p_pca)[0]
        max_prob = np.max(probs)
        cluster_id = np.argmax(probs) + 1
        
        # Define display variables based on outlier status
        if is_atypical:
            theme = "success"
            display_value = "" # Hides the percentage
            status_text = "NOT SIMILAR TO MALIGNANT MORPHOLOGY"
        else:
            # Show percentage for standard clusters
            display_value = f"{max_prob*100:.1f}%"
            status_text = f"SIMILAR TO CLUSTER {cluster_id}"
            
            # Use 'warning' for lower confidence or borderline cases
            if max_prob < 0.50:
                theme = "warning"
            else:
                theme = "pink"
        
        return ui.value_box(
            "Cluster Model 2",
            display_value,
            status_text,
            theme=theme,
            class_="fs-2"
        )

    # --- ENSEMBLE STATUS ---
    @render.ui
    def ensemble_status():
        X = get_patient().drop(columns=['diagnosis'])
        prob_xgb = xgb_model.predict_proba(X)[0][1]
        xgb_m = prob_xgb >= 0.35
        p_pca_g = pca_global.transform(scaler_global.transform(X))
        global_m = gmm_global.predict(p_pca_g)[0] == global_malig_id
        p_pca_m = pca_malig.transform(scaler_malig.transform(X))
        log_prob = gmm_malig.score_samples(p_pca_m)[0]
        max_subtype_conf = np.max(gmm_malig.predict_proba(p_pca_m)[0])
        
        is_uncertain = (log_prob < outlier_threshold) or (max_subtype_conf < 0.50)
        votes = sum([xgb_m, global_m, (not is_uncertain and xgb_m)])
        
        if votes == 3 and max_subtype_conf > 0.75:
            return ui.HTML("<div class='alert' style='background-color: #e377c2; color: white;'>Suggestive of Malignancy</div>")
        elif votes == 0:
            return ui.HTML("<div class='alert alert-success bg-success text-white'>Suggestive of Benign</div>")
        elif is_uncertain:
            return ui.HTML("<div class='alert alert-warning bg-warning text-dark'>CAUTION: Ambiguous Morphological Features</div>")
        else:
            return ui.HTML("<div class='alert alert-info bg-info text-white'>Conflicting Model Signals: Manual Review Required</div>")

    @render.text
    def true_label():
        label = get_patient()['diagnosis'].values[0]
        return "PATHOLOGY CONFIRMED: MALIGNANT" if label == "M" else "PATHOLOGY CONFIRMED: BENIGN"

    # --- PLOTS (Restored Borders & Tableau 10) ---
    @render_widget
    def global_plot():
        X = get_patient().drop(columns=['diagnosis'])
        p_pca = pca_global.transform(scaler_global.transform(X))
        
        fig = px.scatter(df_global_bg, x='pca1', y='pca2', color='Diagnosis', labels={'pca1': 'Primary Morphological Variation', 'pca2': 'Secondary Morphological Variation'},
                         color_discrete_map={'Malignant (M)': TABLEAU10[3], 'Benign (B)': TABLEAU10[0]}, opacity=0.3)
        
        # RESTORED: Convex Hull Boundaries
        for diag_label, color in {'Malignant (M)': TABLEAU10[3], 'Benign (B)': TABLEAU10[0]}.items():
            pts = df_global_bg[df_global_bg['Diagnosis'] == diag_label][['pca1', 'pca2']].values
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                h_pts = np.append(pts[hull.vertices], [pts[hull.vertices][0]], axis=0)
                fig.add_trace(go.Scatter(x=h_pts[:, 0], y=h_pts[:, 1], mode='lines', fill='toself', 
                                         fillcolor=color, opacity=0.1, showlegend=False, hoverinfo='skip'))

        fig.add_trace(go.Scatter(x=[p_pca[0, 0]], y=[p_pca[0, 1]], mode='markers',
                                 marker=dict(size=18, color='black', symbol='diamond', line=dict(color='white', width=2)),
                                 name="Current Patient"))
        fig.update_layout(height=400, margin=dict(l=10, r=10, b=10, t=10), template="simple_white")
        return fig

    @render_widget
    def malig_plot():
        X = get_patient().drop(columns=['diagnosis'])
        p_pca = pca_malig.transform(scaler_malig.transform(X))
        
        fig = px.scatter(df_malig_bg, x='pca1', y='pca2', color='Cluster', labels={'pca1': 'Primary Morphological Variation', 'pca2': 'Secondary Morphological Variation'},
                         color_discrete_sequence=TABLEAU10, opacity=0.5)
        
        # RESTORED: Cluster Boundaries
        for i, cluster in enumerate(sorted(df_malig_bg['Cluster'].unique())):
            pts = df_malig_bg[df_malig_bg['Cluster'] == cluster][['pca1', 'pca2']].values
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                h_pts = np.append(pts[hull.vertices], [pts[hull.vertices][0]], axis=0)
                fig.add_trace(go.Scatter(x=h_pts[:,0], y=h_pts[:,1], mode='lines', fill='toself', 
                                         fillcolor=TABLEAU10[i % 10], opacity=0.15, showlegend=False, hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(x=[p_pca[0,0]], y=[p_pca[0,1]], mode='markers',
                                 marker=dict(size=18, color='black', symbol='diamond', line=dict(color='white', width=2)),
                                 name="Current Patient"))
        fig.update_layout(height=400, margin=dict(l=10,r=10,b=10,t=10), template="simple_white")
        return fig
    
    # --- Feature Importance Output ---
    @render.ui
    def feature_drivers():
        X = get_patient().drop(columns=['diagnosis'])
        
        # 1. XGBoost Top Feature
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        raw_xgb_name = X.columns[indices[0]]
        top_feat_name = translate_feature(raw_xgb_name)
        top_feat_weight = importances[indices[0]]
        
        # 2. GMM "Anomalous" Feature
        X_scaled = scaler_global.transform(X)
        top_deviant_idx = np.argmax(np.abs(X_scaled))
        raw_gmm_name = X.columns[top_deviant_idx]
        top_deviant_name = translate_feature(raw_gmm_name)
        
        return ui.div(
            ui.p(ui.tags.b("Prediction Model: "), top_feat_name, style="font-size: 0.85rem; margin-bottom: 2px;"),
            ui.p(ui.tags.small(f"Contribution Weight: {top_feat_weight*100:.1f}%"), class_="text-muted"),
            ui.p(ui.tags.b("Cluster Model 1: "), top_deviant_name, style="font-size: 0.85rem; margin-bottom: 0;"),
            class_="p-1 border rounded bg-light"
        )
    
    # --- Helper for Feature Translation ---
    def translate_feature(col_name):
        # Mapping of base features
        base_map = {
            "radius": "Radius",
            "texture": "Texture",
            "perimeter": "Perimeter",
            "area": "Area",
            "smoothness": "Smoothness",
            "compactness": "Compactness",
            "concavity": "Concavity",
            "concave points": "Concave Points",
            "symmetry": "Symmetry",
            "fractal dimension": "Fractal Dimension"
        }
    
        name_lower = col_name.lower().replace("_", " ")
        
        # Identify the suffix type
        if "worst" in name_lower:
            suffix = " (Worst)"
        elif "se" in name_lower or "error" in name_lower:
            suffix = " (Standard Error)"
        else:
            suffix = " (Mean)"
            
        # Find the base name and combine
        for key, value in base_map.items():
            if key in name_lower:
                return f"{value}{suffix}"
                
        return col_name.title() # Fallback


app = App(app_ui, server)