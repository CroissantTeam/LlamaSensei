FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

EXPOSE 8081

CMD ["streamlit", "run", "UI.py", "--server.port", "8081"]
