from engine import *
import flask
from flask import Flask, request
import sql_init
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)
current_dir = os.path.dirname(__file__)
predictor = Predictor(path=current_dir+'/models/')
db = os.path.join(current_dir, 'news.sqlite')
label2name = {0: 'truth', 1: 'fake'}


@app.route("/", methods=["GET"])
def general():
    return """Please use http://localhost:8184/ to POST"""


@app.route('/predict', methods=['POST'])
def predict():
    data = {}
    if request.method == 'POST':
        text = ""
        request_json = flask.request.get_json()
        if request_json['text']:
            text = request_json['text']
        label, score = predictor.predict(text)
        score = np.round(score, 4) if score is not None else 'bert'
        data = {"text": text, "label": label, "score": score}
        sql_init.insert_data(db=db, **data)

    return flask.jsonify(data)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8183))
    app.run(host='0.0.0.0', debug=False, port=port)
