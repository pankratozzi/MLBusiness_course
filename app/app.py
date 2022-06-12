from engine import *
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
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


class TextForm(Form):
    text_check = TextAreaField('',
                               [validators.DataRequired(),
                                validators.length(min=predictor.query.get('min'))])


@app.route('/')
def index():
    form = TextForm(request.form)
    return render_template('textform.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    form = TextForm(request.form)
    if request.method == 'POST' and form.validate():
        textform = request.form['text_check']
        label, score = predictor.predict(textform)
        score = np.round(score, 4) if score is not None else 'bert'
        data = {'text': textform, 'label': label, 'score': score}
        sql_init.insert_data(db=db, **data)
        return render_template('results.html',
                               content=textform,
                               prediction=label2name[label],
                               probability=np.round(score*100, 2) if score != 'bert' else score,)
    return render_template('textform.html', form=form)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=False, port=port)
