# Project 01: Support Ticket Classifier (Telecom, 40+ Topics)

## Задача
Автоматическая классификация обращений клиентов телеком-оператора по 40+ тематикам с системой уверенности: тикеты с уверенностью выше порога маршрутизируются автоматически, остальные эскалируются оператору.

## Архитектура
- **Модель**: TF-IDF + LogisticRegression с калибровкой уверенности.
- **Инференс**: Экспорт в ONNX и выполнение через `onnxruntime`.
- **Сервис**: FastAPI microservice (`/classify`).
- **Инфраструктура**: Dockerfile + Kubernetes Deployment (2 реплики, readiness/liveness probes) и Service.

## Целевые метрики
- Macro-F1 на отложенной выборке ≥ 0.80.
- Доля автоматической маршрутизации при пороге confidence ≥ 0.40 составляет ≥ 40% с точностью (precision) ≥ 0.90.
- Полное совпадение результатов инференса ONNX с baseline моделью sklearn.

## Как запустить
```bash
pytest -q tests/test_01_ticket_classifier.py
```

## Принятые компромиссы
- Замена ruBERT на TF-IDF + Logistic Regression обеспечивает мгновенный инференс на CPU, сохраняя полный продакшн-пайплайн (ONNX, FastAPI, Docker, K8s).
