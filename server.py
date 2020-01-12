import os
import pickle
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["POST"])
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
    prediction = clf.predict([[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]])

    return {"result": prediction[0]}

if __name__ == "__main__":
    debug_mode = os.environ["DEBUG_MODE"] == "1"
    http_port = int(os.environ["HTTP_PORT"])
    http_host = os.environ["HTTP_HOST"]
    app.run(debug=debug_mode, host=http_host, port=http_port)