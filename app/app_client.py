import numpy as np
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from requests.exceptions import ConnectionError
import requests


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='secret',
)
label2name = {0: 'truth', 1: 'fake'}


class TextForm(Form):
    text_check = TextAreaField('',
                               [validators.DataRequired(),
                                validators.length(min=63)])


def get_json(text):
    body = {
        'text': text,
    }
    my_url = "http://192.168.0.22:8183/predict"
    headers = {'content-type': 'application/json; charset=utf-8'}
    response = requests.post(my_url, json=body, headers=headers)
    return response.json()


@app.route('/')
def index():
    form = TextForm(request.form)
    return render_template('textform.html', form=form)


@app.route('/results', methods=['GET', 'POST'])
def results():
    form = TextForm(request.form)
    data = dict()
    if request.method == 'POST' and form.validate():
        data['text'] = request.form.get('text_check')
        try:
            response = get_json(data['text'])
            textform = response.get('text')
            label = response.get('label')
            score = response.get('score')
            return render_template('results.html',
                                   content=textform,
                                   prediction=label2name[label],
                                   probability=np.round(score * 100, 2) if score != 'bert' else score, )
        except ConnectionError:
            return render_template('textform.html', form=form)
    return render_template('textform.html', form=form)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8184, debug=False)
