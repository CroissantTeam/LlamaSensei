FROM python:3.10-slim

RUN mkdir /app
COPY . /app/

WORKDIR /app
RUN apt-get update && \
    apt-get install -y git
RUN pip install -r requirements.txt

ENV HOME=/app

CMD ["python", "main.py"]
