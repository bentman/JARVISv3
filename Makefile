# JARVISv3 Development Makefile
# Safe, minimal entrypoints using existing scripts and Docker config

.PHONY: setup backend-dev frontend-dev validate test docker-build docker-up docker-down docker-logs

# Setup: Create venv and install dependencies
setup:
	python -m venv backend/.venv
	backend/.venv/Scripts/python -m pip install --upgrade pip
	backend/.venv/Scripts/python -m pip install -r backend/requirements.txt
	@echo Setup complete. Use 'make validate' to verify installation.

# Development servers
backend-dev:
	backend/.venv/Scripts/python backend/main.py

frontend-dev:
	cd frontend && npm run dev

# Validation and testing
validate:
	backend/.venv/Scripts/python scripts/validate_backend.py

test:
	backend/.venv/Scripts/python -m pytest tests/

# Docker operations
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f
