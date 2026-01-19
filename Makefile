
# Variables
PYTHON := uv run
RUFF := uv run ruff

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make lint     - Run linting and formatting"
	@echo "  make run      - Run the main script (shows help)"
	@echo "  make run-db   - Run the ChromaDB server"
	@echo "  make clean    - Remove artifacts"

.PHONY: install
install:
	uv sync

.PHONY: lint
lint:
	$(RUFF) check . --fix

.PHONY: run
run:
	$(PYTHON) main.py --help

.PHONY: run-db
run-db:
	./run-db.sh

.PHONY: clean
clean:
	rm -rf .ruff_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
