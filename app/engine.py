import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizerFast
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer
import spacy


seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = "C:\\Users\\Daria\\Downloads\\Lection2-20220325T130258Z-001\\Lection2\\"
nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
stop = stopwords.words('english')
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=1024)


def preprocessor(text, stop=stop):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower())
    text = re.sub(r'[^\w]', ' ', text)
    text = [w for w in text.split() if w not in stop]
    text = [re.sub(r"[^a-zA-Z0-9]+", '', k) for k in text]
    return re.sub(' +', ' ', ' '.join(text))


def snow_stem(text):
    snow_stemmer = SnowballStemmer(language='english')
    new_text = ''
    for word in text.split():
        new_text += snow_stemmer.stem(word) + ' '
    return new_text


def lemmatize(text, nlp=nlp):
    new_text = ''
    for token in nlp(text):
        new_text += token.lemma_ + ' '
    return new_text


def preprocessing_for_bert(data: np.ndarray):
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer(snow_stem(preprocessor(sent)), padding='max_length', truncation=True, max_length=1024)
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


class BertModel(nn.Module):
    def __init__(self, tune=False):
        super(BertModel, self).__init__()
        self.bert = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.tune = tune
        for param in self.bert.parameters():
            param.requires_grad = tune
        self.classifier = nn.Sequential(nn.Linear(768, 768),
                                        nn.LeakyReLU(0.01, inplace=True),
                                        nn.Dropout(0.1),
                                        nn.Linear(768, 2)
                                        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stop=None, nlp=None, lemma=True, verbose=False):
        self.stop = stop
        self.nlp = nlp
        if stop is None:
            stop = stopwords.words('english')
        if nlp is None:
            nlp = spacy.load('en_core_web_sm')
        self.lemma = lemma
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        if isinstance(X_, pd.Series) and X_.name == 'title':
            if self.verbose:
                print('Applying title tranformations')
            X_ = X_.apply(lambda x: preprocessor(x, stop=self.stop))
            if self.lemma:
                X_ = X_.apply(lambda x: lemmatize(x, nlp=self.nlp))
            else:
                X_ = X_.apply(lambda x: snow_stem(x))
        elif isinstance(X_, pd.DataFrame) and ('title' in X_.columns):
            if self.verbose:
                print('Applying text tranformations')
            X_['title'] = X_['title'].apply(lambda x: preprocessor(x, stop=self.stop))
            if self.lemma:
                X_['title'] = X_['title'].apply(lambda x: lemmatize(x, nlp=self.nlp))
            else:
                X_['title'] = X_['title'].apply(lambda x: snow_stem(x))
        elif isinstance(X_, pd.DataFrame) and ('text' in X_.columns):
            if self.verbose:
                print('Applying text transformations')
            X_['text'] = X_['text'].apply(lambda x: preprocessor(x, stop=self.stop))
            if self.lemma:
                X_['text'] = X_['text'].apply(lambda x: lemmatize(x, nlp=self.nlp))
            else:
                X_['text'] = X_['text'].apply(lambda x: snow_stem(x))
        elif isinstance(X_, pd.Series) and X_.name == 'text':
            if self.verbose:
                print('Applying title tranformations')
            X_ = X_.apply(lambda x: preprocessor(x, stop=self.stop))
            if self.lemma:
                X_ = X_.apply(lambda x: lemmatize(x, nlp=self.nlp))
            else:
                X_ = X_.apply(lambda x: snow_stem(x))
        else:
            raise ValueError(f'Input name should be "title" or "text", actual name is {X_.name}')
        return X_


class CountLDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5, learning_method='batch', max_features=1000, stop_words='english',
                 n_jobs=-1, random_state=seed):  # max_iter=10 as default
        self.n_components = n_components
        self.learning_method = learning_method
        self.max_features = max_features
        self.stop_words = stop_words
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.pipe = None

    def fit(self, X, y=None):
        self.pipe = make_pipeline(CountVectorizer(stop_words=self.stop_words, max_features=self.max_features),
                                  LatentDirichletAllocation(n_components=self.n_components,
                                                            learning_method=self.learning_method,
                                                            n_jobs=self.n_jobs,
                                                            random_state=self.random_state)).fit(X['text'].values)
        return self

    def transform(self, X):
        X_ = np.zeros((X.shape[0], self.n_components))
        for i in range(X_.shape[0]):
            X_[i, :] = self.pipe.transform(X.values[i])
        return pd.DataFrame(data=X_, columns=[f'text_{i}' for i in range(self.n_components)], index=X.index)


class Predictor:
    def __init__(self, path=''):
        self.bert = BertModel()
        self.bert.load_state_dict(torch.load(path + 'bert_model.pth', map_location=torch.device(device)))
        self.cat = CatBoostClassifier()
        self.cat.load_model(path + 'cat_model')
        self.lgbm = joblib.load(path + '/lda_lgbm.pkl')
        self.thresholds = {'cat': 0.441733, 'lgbm': 0.460132}
        self.query = {'min': 63, 'cat_max': 286, 'upper': 3105}  # else bert

    def predict(self, X: str):
        length = len(X)
        name = 'title' if (length >= self.query.get('min')) and (length <= self.query.get('cat_max')) else 'text'
        X = pd.DataFrame({name: X}, index=[0,])
        if name == 'title':
            label, score = self.cat_predict(X)
            if score < self.thresholds.get('cat'):
                score = 1 - score
            return label, score
        elif name == 'text' and length <= self.query.get('upper'):
            label, score = self.lgbm_predict(X)
            if score < self.thresholds.get('lgbm'):
                score = 1 - score
            return label, score
        else:
            label = self.bert_predict(X)
            return label, None

    def bert_predict(self, X: pd.DataFrame) -> int:
        inp_ids, mask = preprocessing_for_bert(X.values[0])
        inp_ids, mask = inp_ids.to(device), mask.to(device)
        logits = self.bert(inp_ids, mask)
        label = torch.argmax(logits, dim=1).flatten()
        return label.item()

    def cat_predict(self, X: pd.DataFrame) -> tuple:
        score = self.cat.predict_proba(X)[:, 1]
        label = score >= self.thresholds.get('cat')
        return int(label[0]), score[0]

    def lgbm_predict(self, X: pd.DataFrame) -> tuple:
        score = self.lgbm.predict_proba(X)[:, 1]
        label = score >= self.thresholds.get('lgbm')
        return int(label[0]), score[0]


if __name__ == '__main__':
    # model = BertModel()
    # model.load_state_dict(torch.load(PATH + "bert_model.pth", map_location=torch.device('cpu')))
    # cat = CatBoostClassifier()
    # cat.load_model('cat_model')
    # df = pd.read_csv(PATH+'Fake.csv')
    # lgbm = joblib.load('lda_lgbm.pkl')
    # X = pd.DataFrame(data=df['text'].values.reshape(-1, 1), columns=['text'])
    # print(lgbm.predict(X.head(1)))
    # inp_ids, mask = preprocessing_for_bert(X.head(1).values[0])
    # logits = model(inp_ids.to('cpu'), mask.to('cpu'))
    # torch.argmax(logits, dim=1).flatten()
    # pr = Predictor()
    # print(pr.cat)
    print()
