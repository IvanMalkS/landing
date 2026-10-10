FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY projects/ projects/

EXPOSE 8000

CMD ["uvicorn", "projects.01_ticket_classifier.app:app", "--host", "0.0.0.0", "--port", "8000"]
