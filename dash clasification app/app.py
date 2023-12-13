from dash import Dash, html, Output, Input, State, callback
import dash_mantine_components as dmc
import pandas as pd
import joblib

app = Dash(__name__)

display_output = html.Div(
    [
        dmc.Modal(
            title="model predicton",
            id="modal-centered",
            centered=True,
            zIndex=10000,
            children=[dmc.Text(id="output")],
        ),
        dmc.Button("classification", id="modal-centered-button"),
    ]
)

input_data = dmc.Stack(
    children=[
        dmc.TextInput(
            id="input-1",
            label="height",
            type="number",
            style={"width": 200},
            placeholder="enter height",
            value=0,
        ),
        dmc.TextInput(
            id="input-2",
            label="weight",
            type="number",
            style={"width": 200},
            placeholder="enter weight",
            value=0,
        ),
        display_output,
    ],
)

header_data = dmc.Header(
    height=60,
    children=[dmc.Text("Dog vs Cat classification")],
    style={
        "backgroundColor": "#8a4244",
        "font-family": "Arial, sans-serif",
        "textAlign": "center",
        "font-size": "36px",
        "font-weight": "bold",
        "text-transform": "uppercase",
        "color": "#333",
    },
)

app.layout = html.Div(
    [
        dmc.MantineProvider(
            theme={"colorScheme": "dark"},
            children=[
                header_data,
                dmc.Paper(
                    [input_data],
                    p="lg",
                ),
            ],
        )
    ],
    style={
        "width": "50%",
        "height": "300px",
        "textAlign": "center",
        "margin": "10px",
        "position": "absolute",
        "top": "30%",
        "left": "30%",
        "margin": "-50px 0 0 -50px",
    },
)


@callback(
    Output("modal-centered", "opened"),
    Output("output", "children"),
    [Input("modal-centered-button", "n_clicks")],
    [
        State("modal-centered", "opened"),
        State("input-1", "value"),
        State("input-2", "value"),
    ],
    prevent_initial_call=True,
)
# def toggle_modal(n_clicks, opened, data1, data2):
#     return not opened , f'{data1} {data2}'


def get_predictions(n_clicks, opened, data1, data2):
    # Unpickle classifier
    clf = joblib.load("clf.pkl")
    # Get values through input bars
    height = data1
    weight = data2
    print(f"height:  {height},\nweight:  {weight}")
    # Put inputs to dataframe
    X = pd.DataFrame([[height, weight]], columns=["Height", "Weight"])
    # Get prediction
    prediction = clf.predict(X)[0]
    return not opened, f"prediction: {prediction}"


if __name__ == "__main__":
    app.run_server(debug=False, host="127.0.0.1", port=8181)
