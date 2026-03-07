import pandas as pd
import math
import plotly.express as px
from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget

# --- 1. DATA SETUP ---
base_df = pd.read_csv('data/data.csv')

# Dynamically calculate the min and max for your sliders
rad_min, rad_max = math.floor(base_df['radius_mean'].min()), math.ceil(base_df['radius_mean'].max())
tex_min, tex_max = math.floor(base_df['texture_mean'].min()), math.ceil(base_df['texture_mean'].max())
area_min, area_max = math.floor(base_df['area_mean'].min()), math.ceil(base_df['area_mean'].max())

# --- 2. UI ARCHITECTURE ---
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Clinical Explorer",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Clinical Biomarkers"),
                ui.p("Adjust the ranges to filter the patient cohort:"),
                
                ui.input_slider("filter_radius", "Radius (Mean)", min=rad_min, max=rad_max, value=[rad_min, rad_max], step=0.5),
                ui.input_slider("filter_texture", "Texture (Mean)", min=tex_min, max=tex_max, value=[tex_min, tex_max], step=0.5),
                ui.input_slider("filter_area", "Area (Mean)", min=area_min, max=area_max, value=[area_min, area_max], step=10.0),
                
                ui.input_action_button("reset_btn", "Reset Filters", class_="btn-success w-100 mt-3"),
                title="Filters"
            ),
        

            # --- KPI CARDS ROW ---
            ui.layout_columns(
                ui.card(
                    ui.card_header("Total Patients", class_="bg-dark text-white"),
                    ui.output_text("kpi_total"),
                    class_="text-center fs-2"
                ),
                ui.card(
                    # Using inline CSS to match the Plotly 'M' red color
                    ui.card_header("Malignant Cases", class_="text-white", style="background-color: #ef553b;"),
                    ui.output_text("kpi_malignant"),
                    class_="text-center fs-2"
                ),
                ui.card(
                    # Using inline CSS to match the Plotly 'B' blue color
                    ui.card_header("Benign Cases", class_="text-white", style="background-color: #636efa;"),
                    ui.output_text("kpi_benign"),
                    class_="text-center fs-2"
                ),
            ),
            
            # --- VISUALIZATION ---
            ui.card(
                ui.card_header("Patient Feature Space (Pre-Clustering)"),
                output_widget("cluster_scatter"),
                full_screen=True, # Pro-tip: This allows judges to expand the plot to full screen!
            ),
            fillable_mobile=True,
        )
    ),
    title=ui.div(
        ui.h1("Breast Cancer Diagnostics", class_="mb-0 fs-4 text-white"),
        class_="d-flex align-items-center",
    ),
    id="navbar",
    navbar_options=ui.navbar_options(
        theme="dark",
        class_="bg-dark text-white p-3 mb-0",
    ),
)

# --- 3. SERVER LOGIC ---
def server(input, output, session):

    @reactive.calc
    def filtered_data():
        df = base_df.copy()

        rad_min, rad_max = input.filter_radius()
        tex_min, tex_max = input.filter_texture()
        area_min, area_max = input.filter_area()

        df = df[
            (df["radius_mean"].between(rad_min, rad_max)) &
            (df["texture_mean"].between(tex_min, tex_max)) &
            (df["area_mean"].between(area_min, area_max))
        ]
        
        return df

    # --- KPI Renders ---
    @render.text
    def kpi_total():
        return str(len(filtered_data()))

    @render.text
    def kpi_malignant():
        df = filtered_data()
        return str(len(df[df["diagnosis"] == "M"]))

    @render.text
    def kpi_benign():
        df = filtered_data()
        return str(len(df[df["diagnosis"] == "B"]))

    # --- Plotly Scatter Chart Render ---
    @render_widget
    def cluster_scatter():
        df = filtered_data()
        
        if df.empty:
            return px.scatter(title="No patients match these filters")

        # Create an interactive scatter plot using Plotly Express
        fig = px.scatter(
            df, 
            x="radius_mean", 
            y="texture_mean", 
            color="diagnosis",
            color_discrete_map={"M": "#ef553b", "B": "#636efa"},
            hover_data=["area_mean", "smoothness_mean"], # Shows extra info when you mouse over a dot
            labels={
                "radius_mean": "Mean Radius",
                "texture_mean": "Mean Texture",
                "diagnosis": "Diagnosis"
            },
            title="Radius vs. Texture"
        )
        
        # Make the dots a bit larger and slightly transparent to see dense clusters
        fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
        
        return fig

    # --- Reset Button Logic ---
    @reactive.effect
    @reactive.event(input.reset_btn)
    def reset_filters():
        ui.update_slider("filter_radius", value=[rad_min, rad_max])
        ui.update_slider("filter_texture", value=[tex_min, tex_max])
        ui.update_slider("filter_area", value=[area_min, area_max])

app = App(app_ui, server)