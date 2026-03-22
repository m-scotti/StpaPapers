FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip show flask  # ← will fail the build loudly if flask isn't installed

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Force Docker to re-copy everything on each build
ARG CACHE_BUST=1
RUN echo "Cache bust: $CACHE_BUST"

COPY . .

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
