FROM python:3.9
LABEL maintainer="pankratozzi@gmail.com"
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8183
EXPOSE 8184
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1H871EerhJeF1Wy6VX5Zv-u1lMfDV4gek' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1H871EerhJeF1Wy6VX5Zv-u1lMfDV4gek" -O bert_model.zip && rm -rf /tmp/cookies.txt
RUN unzip bert_model.zip
RUN rm -rf bert_model.zip
RUN mv bert_model.pth /app/app/models
COPY ./docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]