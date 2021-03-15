FROM python:3.7.6

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

CMD ['python', 'server.py']