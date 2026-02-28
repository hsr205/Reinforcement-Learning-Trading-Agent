# Agentic Stock Trader

<p align="center">
  An agent-based reinforcement learning system for intelligent stock prediction and decision-making.
  <br/>
  Built with structured configuration, reproducible environments, and research-backed methodology.
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/hsr205/agentic_stock_predictor?style=for-the-badge" />
  <img src="https://img.shields.io/github/last-commit/hsr205/agentic_stock_predictor?style=for-the-badge" />
  <img src="https://img.shields.io/github/license/hsr205/agentic_stock_predictor?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Poetry-Dependency%20Management-blueviolet?style=for-the-badge" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Initialization](#initialization)
- [Environment Set-Up](#environment-set-up)
  - [Configuration Management](#configuration-management)
  - [Environment Variables](#environment-variables)
  - [Configuration Module Structure](#configuration-module-structure)
- [Application Execution](#application-execution)
- [Exporting Historical Stock Data](#exporting-historical-stock-data)
  - [Equities To Follow](#equities-to-follow)
- [Sources](#sources)
- [Conclusions](#conclusions)

---

## Overview

**ADD OVERVIEW SECTION MATERIAL**

---

## Initialization

This project uses **Poetry** for dependency management and virtual environment handling.

Poetry provides:
- Deterministic dependency resolution via `poetry.lock`
- Centralized configuration via `pyproject.toml`
- Automatic virtual environment management
- Separation of runtime and development dependencies

### 1. Installing Poetry (Mac)

Install Poetry using the official installer:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify installation:

```bash
poetry --version
```

---

### 2. Initializing Poetry in the Project

If this project has not yet been initialized with Poetry, run:

```bash
poetry init
```

This creates a `pyproject.toml` file containing project metadata and dependency definitions.

---

### 3. Installing Project Dependencies

After cloning the repository, install dependencies:

```bash
poetry install
```

This creates a virtual environment and installs all dependencies listed in `pyproject.toml` according to the locked versions in `poetry.lock`.

---

### 4. Adding Dependencies

Add a runtime dependency:

```bash
poetry add <library_name>
```

Example:

```bash
poetry add numpy
```

---

## Environment Set-Up

### Configuration Management

This project uses a centralized configuration module located at:

```
config/config.py
```

The configuration system is built using **Pydantic Settings** and is responsible for:

- Loading environment variables from a `.env` file
- Validating required configuration values
- Enforcing strong typing
- Failing fast if required variables are missing

---

### Environment Variables

A `.env` file must exist in the project root:

```
Final_Project_Materials/.env
```

Example:

```
APP_NAME=AGENTIC_STOCK_PREDICTOR
```

The `Settings` class inside `config.py` defines all application configuration parameters.

The settings object is instantiated once:

```python
settings = Settings()
```

It can then be imported anywhere in the project:

```python
from config.config import settings
```

This ensures:

- Centralized configuration
- No duplicated environment reads
- Strong validation guarantees
- Clean separation of configuration from application logic

---

## Application Execution

The entry point of the application is:

```
main/main.py
```

To execute the application correctly, it must be run as a module from the project root.

---

### Required Project Structure

Ensure the following files exist:

```
main/__init__.py
config/__init__.py
```

These files mark the directories as Python packages and allow proper module imports.

---

### Run the Application (Recommended Method)

From the project root:

```bash
poetry run python -m main.main
```

This ensures:

- The correct Poetry-managed virtual environment is used
- Module imports resolve correctly
- The project root is added to the Python path

---

## Exporting Historical Stock Data

## Purpose

Before training any reinforcement learning agent on financial markets, we must first collect clean, structured historical data.

In reinforcement learning, the environment is defined by how the system evolves over time. For a trading agent, this evolution is driven by historical price movements.

This method exports minute-level stock data for an entire year into CSV files so it can later be used to construct a reinforcement learning environment.

Exporting to CSV ensures:

- Reproducibility  
- Separation between data collection and training  
- Compatibility with pandas, NumPy, PyTorch, and other ML tools  

---

## What the Method Does

The method:

```python
export_historical_stock_data(self, year_of_data_to_collect: int)
```

For each ticker symbol:

1. Iterates through all 12 months of the specified year  
2. Requests minute-level stock bars  
3. Cleans and structures the returned data  
4. Aggregates all months into one yearly DataFrame  
5. Exports the final dataset to a local CSV file  

Each ticker produces one CSV file containing a full year of minute-resolution data.

---

This export step forms the foundation for constructing a trading Markov Decision Process and training a reinforcement learning agent.

---

### Equities To Follow

Throughout the course of our project we will be following the intraday price movements of the following equities:

![TSLA](https://img.shields.io/badge/TSLA-Tesla-CC0000?logo=tesla&logoColor=white)

![AAPL](https://img.shields.io/badge/AAPL-Apple-black?logo=apple)

![META](https://img.shields.io/badge/META-Meta-0467DF?logo=meta&logoColor=white)

![AMZN](https://img.shields.io/badge/AMZN-Amazon-FF9900?logo=amazon&logoColor=black)

![MSFT](https://img.shields.io/badge/MSFT-Microsoft-5E5E5E?logo=microsoft&logoColor=white)

![NVDA](https://img.shields.io/badge/NVDA-NVIDIA-76B900?logo=nvidia&logoColor=white)

![GOOGL](https://img.shields.io/badge/GOOGL-Alphabet-4285F4?logo=google&logoColor=white)

---

## Sources

**List all references used in the project.**

1. Soumo Chatterjee (2024). [7 applications of reinforcement learning in finance and trading](https://neptune.ai/blog/7-applications-of-reinforcement-learning-in-finance-and-trading)

2. Junhao Zhang, Yifei Lei (2022). [Deep Reinforcement Learning for Stock Prediction](https://onlinelibrary.wiley.com/doi/10.1155/2022/5812546)

3. Yasin, A. S., & Gill, P. S. (2024, November). [Reinforcement learning framework for quantitative trading](https://arxiv.org/html/2411.07585v1) 

4. Ziyi (Queena) Zhou, Nicholas Stern, Julien Laasri (2025). [Reinforcement Learning for Stock Transactions](https://arxiv.org/html/2505.16099v2)

**Format:**

- Author. (Year). *Title*. Source. Link (if applicable)

---

## Conclusions

**Summarize findings and outcomes.**

Include:
- Key results
- Performance metrics
- Limitations
- Future improvements
- Lessons learned

Example:
- Model accuracy:
- Major insight:
- Bottlenecks:
- Next steps:

---
