FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY policylora ./policylora
RUN pip install --no-cache-dir .

ENV DETECTOR=mock
EXPOSE 8000
CMD ["uvicorn", "policylora.serve.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
