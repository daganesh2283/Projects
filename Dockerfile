# ==========================================
# Stage 1: Build the React Frontend
# ==========================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend-react

# Copy frontend source files
COPY frontend-react/package*.json ./
RUN npm ci

COPY frontend-react/ ./
RUN npm run build

# ==========================================
# Stage 2: Build the FastAPI Backend & Serve
# ==========================================
FROM python:3.11-slim AS backend-runner
WORKDIR /app

# Install system dependencies (needed for compiling certain python packages if necessary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files and outputs stage template
COPY api.py main.py generate_pdf_report.py app.py ./
COPY src/ ./src
COPY data/ ./data
COPY synthetic_data/ ./synthetic_data

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend-react/dist ./frontend-react/dist

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Command to run uvicorn server
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
