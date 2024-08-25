FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

RUN mkdir /app
COPY . /app/

WORKDIR /app
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt

ENV HOME=/app

RUN yt-dlp --username oauth2 --password '' https://www.youtube.com/watch?v=sXrasaDZxw0

CMD ["python", "main.py"]
