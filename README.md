# mlds-400-titanic-disaster

A Dockerized machine learning project that predicts passenger survival on the Titanic using logistic regression models in both Python and R.

---

## Table of Contents

* [Overview](#overview)
* [Repository Structure](#repository-structure)
* [Prerequisites](#prerequisites)
* [Data Setup](#data-setup)
* [Running the Python Pipeline](#running-the-python-pipeline)
* [Running the R Pipeline](#running-the-r-pipeline)
* [Output](#output)

---

## Overview

This project implements logistic regression models to predict Titanic passenger survival.

---

## Repository Structure

```

.
├── src/
│   ├── data/                    # Place dataset files here
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── results/             # Output predictions saved here
│   ├── app-python/
│   │   └── app.py
│   └── app-r/
│       ├── app.R
│       └── install_packages.R
├── .dockerignore
├── .gitignore
├── Dockerfile
├── Dockerfile.R
├── README.md
└── requirements.txt
```

---

## Prerequisites

* **Docker** installed and running ([Download Docker](https://www.docker.com/products/docker-desktop))
* **Terminal/Command Line** access (PowerShell on Windows, Terminal on macOS/Linux)
* **Git** (to clone the repository)

---

## Data Setup

1. **Download the Titanic dataset** from: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data?select=gender_submission.csv)
2. **Extract and place files** in the `src/data/` directory:
   * `train.csv` → `src/data/train.csv`
   * `test.csv` → `src/data/test.csv`
3. **Verify structure:**
   ```
   src/data/train.csv
   src/data/test.csv
   ```

---

## Running the Python Pipeline

### Step 1: Build the Docker Image

Navigate to the repository root and run:

```bash
docker build -f src/app-python/Dockerfile -t titanic-python-app .
```

### Step 2: Run the Container

**On Windows (PowerShell):**

```powershell
docker run --rm -v "${PWD}/src/data:/app/src/data" titanic-python-app
```

**On macOS/Linux:**

```bash
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-python-app
```

### Step 3: View Results

The terminal will display:

* **CHANGES:** Data preprocessing steps applied
* **RESULTS:** Training accuracy

Predictions are saved to: `src/data/results/test_predictions_py.csv`

---

## Running the R Pipeline

### Step 1: Build the Docker Image

```bash
docker build -f src/app-r/Dockerfile -t titanic-r-app .
```

### Step 2: Run the Container

**On Windows (PowerShell):**

```powershell
docker run --rm -v "${PWD}/src/data:/app/src/data" titanic-r-app
```

**On macOS/Linux:**

```bash
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-r-app
```

### Step 3: View Results

The terminal will display:

* **CHANGES:** Data preprocessing steps applied
* **RESULTS:** Training accuracy

Predictions are saved to: `src/data/results/test_predictions.csv`

---

## Output

Both pipelines produce:

1. **Terminal Output:**
   * Data preprocessing steps
   * Model training accuracy on the training set
2. **CSV Files:**
   * Python: `src/data/results/test_predictions_py.csv`
   * R: `src/data/results/test_predictions.csv`

The `results/` directory is created automatically if it doesn't exist.

---

## Author

**Yifei Hang**

Northwestern University MLDS

Introduction to Data Engineering
