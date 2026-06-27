FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install .

ENV PYTHONPATH=/app/src
ENV PORT=8080

EXPOSE 8080

CMD ["python", "-m", "nflreadpy.ui", "--host", "0.0.0.0", "--port", "8080"]