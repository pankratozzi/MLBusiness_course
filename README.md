Простой проект по предсказанию достоверности текста новости. Новость о политике США в период президентства Дональда Трампа. 
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

#### Модель ML состоит из трех частей. 
* CatBoostClassifier с text_features для предсказания заголовков (применяется также для коротких тезисных текстов)
* Pipeline(CountVectorizer, LatentDirichletAllocation, LGBMClassifier) преобразует длинную статью в вектор из тем статьи, а затем классифицирует данный образец.
* PyTorch Bert модель для токенизации длинных текстов и их классификации

Для небольших статей используется LGBMClassifier, для длинных модель Bert (HuggingFace), что несколько избыточно, но было принято решение реализовать несколько моделей для расширения возможностей приложения.

Предложено два варианта: 
* app.py - сервер, к которому можно подключиться и получать предсказания
* app_server.py, app_client.py отдельно сервер, получающий запрос и возвращающий словарь с результатами и клиент, который запускает визуализацию и взаимодействует с сервером.
#### Для установки:
```
$ git clone https://github.com/pankratozzi/MLBusiness_course.git
$ cd MLBusiness_course
$ docker build -t fake_news_detector .
```
#### Затем запускаем образ.
```
$ docker run -dp 8183:8183 -p 8184:8184 fake_news_detector
```

#### Можно также (предварительно скачав веса нейронной сети https://www.kaggle.com/datasets/pankratozzi/bert-weights):
```
$ git clone https://github.com/pankratozzi/MLBusiness_course.git
$ pip install -r requirements.txt
```
разместить загруженные веса в директории /models и запустить главный скрипт:
 ```
$ python app.py
 ```
Или скрипты:
```
$ python app_server.py
$ python app_client.py
```
Перейти по ссылке _http://localhost:8184_ на запущенный на локальной машине интерфейс.
Далее в текстовую форму вносим текст для классификации, получаем предсказание с указанием уровня уверенности (кроме Bert модели) модели.

**В целом, модель решает данную узкую (и, кстати, довольно субъективную) задачу.**
