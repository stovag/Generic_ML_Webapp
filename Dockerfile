FROM python:3.10.11-slim-bullseye
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . .
CMD ["streamlit", "run", "app.py"]

