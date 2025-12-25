# -------- Whisper.cpp build stage --------
FROM debian:stable-slim AS whisper_builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN git clone https://github.com/ggerganov/whisper.cpp.git
WORKDIR /opt/whisper.cpp
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j


# -------- llama.cpp build stage --------
FROM debian:stable-slim AS llama_builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /opt/llama.cpp
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
 && cmake --build build -j


# -------- Piper TTS build stage --------
FROM debian:stable-slim AS piper_builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN git clone https://github.com/rhasspy/piper.git
WORKDIR /opt/piper
# Build Piper C++ inference engine
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j


# -------- Production runtime stage --------
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Set work directory
WORKDIR ${APP_HOME}

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        postgresql-client \
        build-essential \
        libpq-dev \
        libffi-dev \
        libssl-dev \
        curl \
        wget \
        libsndfile1 \
        espeak-ng \
        espeak-ng-data \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy AI model executables and libraries from build stages
COPY --from=whisper_builder /opt/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper
COPY --from=whisper_builder /opt/whisper.cpp/build/*.so* /usr/local/lib/

COPY --from=llama_builder /opt/llama.cpp/build/bin/ /usr/local/bin/

COPY --from=piper_builder /opt/piper/build/piper /usr/local/bin/piper

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
