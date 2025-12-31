FROM python:3.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements (Renaming it to .txt inside container automatically)
COPY ./requirements /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy backend and frontend
COPY ./backend /code/backend
COPY ./frontend /code/frontend

# --- THE MAGIC FIX ---
# Copy the model folder from your root into 'backend/model' inside the container
COPY ./model /code/backend/model

# Setup permissions
RUN mkdir -p /code/cache
ENV MPLCONFIGDIR=/code/cache
RUN chmod -R 777 /code/cache

# Start the app
CMD ["gunicorn", "backend.app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]