import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# === Charger les données avec gestion d’erreur ===
file_path = r"C:\Users\Aymen\Desktop\projet py\Sales\sales_data_sample.csv"
try:
    df = pd.read_csv(file_path, encoding='latin1')
except Exception as e:
    df = pd.DataFrame()
    error_msg = f"Erreur de chargement du fichier : {e}"
else:
    error_msg = None

# Normaliser les colonnes si le fichier est chargé
if not df.empty:
    df.columns = [c.strip().upper() for c in df.columns]
    # Sélectionner les colonnes utiles
    cols = ['ORDERDATE', 'COUNTRY', 'PRODUCTLINE', 'SALES', 'QUANTITYORDERED']
    df = df[[c for c in cols if c in df.columns]].copy()
    # Convertir les types
    df['ORDERDATE'] = pd.to_datetime(df.get('ORDERDATE'), errors='coerce')
    df['SALES'] = pd.to_numeric(df.get('SALES'), errors='coerce')
    df['QUANTITYORDERED'] = pd.to_numeric(df.get('QUANTITYORDERED'), errors='coerce')
    # Calculer le profit si absent
    if 'PROFIT' not in df.columns:
        df['PROFIT'] = df['SALES'] * 0.1
    else:
        df['PROFIT'] = pd.to_numeric(df.get('PROFIT'), errors='coerce')

app = Dash(__name__)
app.title = "Tableau de bord des ventes"

app.layout = html.Div([
    html.H1("📊 Tableau de bord des ventes", style={"textAlign": "center"}),
    html.Div(id="error-section"),
    html.Div([
        html.Div([
            html.Label("Région :"),
            dcc.Dropdown(
                options=[{"label": "Toutes", "value": "Toutes"}] +
                        [{"label": r, "value": r} for r in sorted(df['COUNTRY'].dropna().unique())] if not df.empty else [],
                value="Toutes",
                id="region-filter"
            )
        ], style={"width": "45%", "display": "inline-block", "marginRight": "5%"}),
        html.Div([
            html.Label("Produit :"),
            dcc.Dropdown(
                options=[{"label": "Tous", "value": "Tous"}] +
                        [{"label": p, "value": p} for p in sorted(df['PRODUCTLINE'].dropna().unique())] if not df.empty else [],
                value="Tous",
                id="product-filter"
            )
        ], style={"width": "45%", "display": "inline-block"})
    ], style={"marginBottom": "30px"}),
    html.Div(id="kpi-section", style={"display": "flex", "justifyContent": "space-around", "marginBottom": "40px"}),
    html.Div([
        html.Div(dcc.Graph(id="sales-trend"), style={"width": "48%", "display": "inline-block"}),
        html.Div(dcc.Graph(id="sales-region"), style={"width": "48%", "display": "inline-block"})
    ]),
    html.Div([
        html.H3("Top 5 produits", style={"textAlign": "center"}),
        dcc.Graph(id="top-products")
    ]),
    html.Div([
        html.H3("Unités vendues par région", style={"textAlign": "center"}),
        dcc.Graph(id="units-region")
    ])
])

@app.callback(
    [Output("error-section", "children"),
     Output("kpi-section", "children"),
     Output("sales-trend", "figure"),
     Output("sales-region", "figure"),
     Output("top-products", "figure"),
     Output("units-region", "figure")],
    [Input("region-filter", "value"),
     Input("product-filter", "value")]
)
def update_dashboard(selected_region, selected_product):
    if df.empty:
        return [
            html.Div([
                html.H2("Erreur de chargement", style={"color": "red", "textAlign": "center"}),
                html.P(error_msg or "Le fichier est vide ou mal formaté.", style={"textAlign": "center"})
            ]),
            [], {}, {}, {}, {}
        ]
    dff = df.copy()
    if selected_region != "Toutes":
        dff = dff[dff['COUNTRY'] == selected_region]
    if selected_product != "Tous":
        dff = dff[dff['PRODUCTLINE'] == selected_product]

    # KPIs
    total_sales = dff['SALES'].sum()
    total_profit = dff['PROFIT'].sum()
    total_units = dff['QUANTITYORDERED'].sum()
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

    kpis = [
        html.Div([
            html.H4("Chiffre d'affaires"),
            html.H2(f"{total_sales:,.0f} €", style={"color": "#6a1b9a"})
        ]),
        html.Div([
            html.H4("Profit total"),
            html.H2(f"{total_profit:,.0f} €", style={"color": "#4caf50"})
        ]),
        html.Div([
            html.H4("Unités vendues"),
            html.H2(f"{total_units:,.0f}", style={"color": "#ff9800"})
        ]),
        html.Div([
            html.H4("Marge bénéficiaire"),
            html.H2(f"{profit_margin:.1f} %", style={"color": "#2196f3"})
        ])
    ]

    # Graphique 1 : Évolution mensuelle
    dff_month = dff.groupby(dff['ORDERDATE'].dt.to_period("M")).agg({
        'SALES': "sum",
        'PROFIT': "sum"
    }).reset_index()
    dff_month['ORDERDATE'] = dff_month['ORDERDATE'].dt.to_timestamp()
    fig_trend = px.line(
        dff_month,
        x='ORDERDATE', y=['SALES', 'PROFIT'],
        labels={'ORDERDATE': "Mois", "value": "Montant (€)"},
        title="Évolution des ventes et profits"
    )
    fig_trend.update_traces(mode="lines+markers")

    # Graphique 2 : Répartition par région
    fig_region = px.pie(
        dff.groupby('COUNTRY')['SALES'].sum().reset_index(),
        names='COUNTRY', values='SALES',
        title="Répartition des ventes par région"
    )

    # Graphique 3 : Top produits
    fig_top = px.bar(
        dff.groupby('PRODUCTLINE')[['SALES', 'QUANTITYORDERED']].sum()
          .sort_values(by='SALES', ascending=False)
          .head(5)
          .reset_index(),
        x='PRODUCTLINE', y=['SALES', 'QUANTITYORDERED'],
        barmode="group",
        title="Top 5 des produits"
    )

    # Graphique 4 : Unités vendues par région
    fig_units_region = px.pie(
        dff.groupby('COUNTRY')['QUANTITYORDERED'].sum().reset_index(),
        names='COUNTRY', values='QUANTITYORDERED',
        title="Unités vendues par région"
    )

    return "", kpis, fig_trend, fig_region, fig_top, fig_units_region

if __name__ == "__main__":
    app.run(debug=True)
