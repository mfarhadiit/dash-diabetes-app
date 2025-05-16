import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('diabetes.csv')

# Prepare model
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Start Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
server = app.server

# Input form fields
input_fields = [
    {'id': 'Pregnancies', 'label': 'Pregnancies'},
    {'id': 'Glucose', 'label': 'Glucose'},
    {'id': 'BloodPressure', 'label': 'Blood Pressure'},
    {'id': 'SkinThickness', 'label': 'Skin Thickness'},
    {'id': 'Insulin', 'label': 'Insulin'},
    {'id': 'BMI', 'label': 'BMI'},
    {'id': 'DiabetesPedigreeFunction', 'label': 'Diabetes Pedigree Function'},
    {'id': 'Age', 'label': 'Age'}
]

# Layout
app.layout = dbc.Container([
    html.H1("🧬 Diabetes Prediction Dashboard", className="text-center text-info my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Patient Information"),
                dbc.CardBody([
                    *[dbc.Input(id=f['id'], placeholder=f['label'], type='number', className='mb-2') for f in input_fields],
                    dbc.Button("Predict", id='predict-btn', color='primary', className='mt-2 w-100'),
                ])
            ])
        ], md=4),

        dbc.Col([
            html.H5("Insights & Visualizations", className="text-white"),
            dcc.Tabs([
                dcc.Tab(label='Age Histogram', children=[
                    dcc.Graph(figure=px.histogram(df, x='Age', color='Outcome', barmode='overlay', nbins=30))
                ]),
                dcc.Tab(label='BMI vs Glucose', children=[
                    dcc.Graph(figure=px.scatter(df, x='BMI', y='Glucose', color='Outcome'))
                ]),
                dcc.Tab(label='BMI Box Plot', children=[
                    dcc.Graph(figure=px.box(df, x='Outcome', y='BMI', color='Outcome'))
                ]),
                dcc.Tab(label='Diabetes Ratio Pie', children=[
                    dcc.Graph(figure=px.pie(df, names='Outcome', title='Diabetic vs Non-Diabetic'))
                ]),
            ])
        ], md=8),
    ]),

    html.Hr(),

    html.H4("📋 Data Sample Preview", className="text-white mt-3"),
    dbc.Table.from_dataframe(df.sample(10), striped=True, bordered=True, hover=True, className='text-white bg-dark'),

    # Modal Alert
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("🧠 Prediction Result"), close_button=True),
        dbc.ModalBody(id='modal-body', className="fs-5 fw-semibold"),
    ], id='result-modal', is_open=False, centered=True, size="lg", backdrop='static'),
], fluid=True)


# Prediction callback
@app.callback(
    Output('modal-body', 'children'),
    Output('result-modal', 'is_open'),
    Input('predict-btn', 'n_clicks'),
    [State(f['id'], 'value') for f in input_fields]
)
def predict(n_clicks, *values):
    if not n_clicks:
        return "", False
    if None in values:
        return "⚠️ Please fill in all fields to proceed with prediction.", True

    input_array = scaler.transform([values])
    pred = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0][1]
    percent = round(prob * 100, 2)

    if prob >= 0.5:
        return (
            f"⚠️ High risk of diabetes detected.\n\n"
            f"Your estimated probability is {percent}%. Please consult with a healthcare professional.",
            True
        )
    else:
        return (
            f"✅ Low risk of diabetes.\n\n"
            f"Your estimated probability is {percent}%. Keep maintaining a healthy lifestyle!",
            True
        )

# Run
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=10000)

