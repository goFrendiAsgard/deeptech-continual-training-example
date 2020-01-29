import io
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, Response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__, static_url_path="", static_folder="static")


@app.route("/", methods=["GET"])
def main_index():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def main_predict():
    json_request = request.get_json()

    model_name = "latest"
    sepal_length = 0.0
    sepal_width = 0.0
    petal_length = 0.0
    petal_width = 0.0
    if "model" in json_request:
        model_name = json_request["model"]
    if "sepal_length" in json_request:
        sepal_length = json_request["sepal_length"]
    if "sepal_width" in json_request:
        sepal_width = json_request["sepal_width"]
    if "petal_length" in json_request:
        petal_length = json_request["petal_length"]
    if "petal_width" in json_request:
        petal_width = json_request["petal_width"]

    clf = pickle.load(open("./model/" + model_name + ".model", "rb"))
    prediction = clf.predict([[
        float(sepal_length), float(sepal_width),
        float(petal_length), float(petal_width)
    ]])

    return {"result": prediction[0]}


@app.route("/accuracy-history")
def main_accuracy_history():
    df = pd.read_csv(
        "./accuracy/log.csv", header=None,
        names=["time", "accuracy"]
    )
    df = df.tail(10)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df["time"], df["accuracy"], "-")
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_title("Accuracy")
    ax.set_ylim([0, 1])
    ax.grid(True)

    fig.tight_layout()

    canvas = FigureCanvas(fig)
    png_output = io.BytesIO()
    canvas.print_png(png_output)
    response = Response(png_output.getvalue(), mimetype="image/png")
    return response


if __name__ == "__main__":
    debug_mode = os.environ["DEBUG_MODE"] == "1"
    http_port = int(os.environ["HTTP_PORT"])
    http_host = os.environ["HTTP_HOST"]
    app.run(debug=debug_mode, host=http_host, port=http_port)
