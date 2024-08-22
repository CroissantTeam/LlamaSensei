FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

RUN mkdir /app
COPY . /app/

WORKDIR /app
RUN apt-get update && \
    apt-get install -y git
RUN pip install -r requirements.txt

ENV HOME=/app

CMD ["python", "main.py"]
