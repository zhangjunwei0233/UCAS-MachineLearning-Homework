This is the homework repo for UCAS "计算机科学导论" course

# Quick start

## Prerequisites

- Python >=3.12
- [uv](https://docs.astral.sh/uv/) package manager

## Setup Environment

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and navigate to the repository**:
   ```bash
   git clone <repository-url>
   cd MachineLearning
   ```

3. **Create and activate virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   uv sync
   ```

## Running Exercises

Each exercise is stored in a seperate folder, with a main.py as entry point

To run any of these exercises, just activate the venv, cd into its dir and run main.py. For example:

```bash
source .venv/bin/activate
cd ex1-house-price-prediction
python main.py
```
