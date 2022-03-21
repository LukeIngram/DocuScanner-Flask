FROM python:3.10.2-stretch

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt 

CMD ["uwsgi","app.ini"]

