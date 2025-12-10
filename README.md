# Optimizing Andalusia’s Sentinel Surveillance Network: A Capacitated P-Center Approach

This repository contains a complete framework for solving the **Capacitated P-Center Problem**, specifically applied to the design of a **Sentinel Surveillance Network** for respiratory viruses in Andalusia, Spain.

The project goes beyond a simple solver; it includes a full data processing pipeline, a multi-objective evolutionary framework, a comparative experimentation suite, and an interactive decision-support dashboard.

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Data Pipeline](#-data-pipeline)
- [Algorithms & Logic](#-algorithms--logic)
- [Installation](#-installation)
- [Usage](#-usage)
    - [Interactive Dashboard](#interactive-dashboard)
    - [Algorithm Demo](#algorithm-demo)
    - [Running Experiments](#running-experiments)
- [Authors](#-authors)

---

## 🔭 Project Overview

The goal is to select an optimal set of $p$ hospitals from a candidate list to minimize the maximum travel time (or distance) for any citizen to their assigned hospital. This is a **Minimax** problem known as the **P-Center Problem**.

**Constraints:**
1.  **Capacity**: Each hospital has a maximum capacity (beds/resources).
2.  **Demand**: Each municipality has a demand based on its population and an infection rate ($\rho$).
3.  **Coverage**: All demand must be satisfied.

We solve this using **ACME (Acceleration by Clustering and Memetic Exploitation)**, which combines global search (Genetic Algorithms) with local refinement (Local Search & Clustering).

---

## 📂 Repository Structure

```text
├── ACME/                   # Educational Demo of the Algorithm
│   └── acme_demo.py        # Generates a GIF visualizing ACME on a continuous function
├── app/                    # Streamlit Web Application
│   ├── app.py              # Main dashboard entry point
│   ├── map_generator.py    # Advanced Folium map rendering (layers, styling)
│   └── cache_manager.py    # Caching system for solutions
├── data/                   # Data Storage
│   ├── raw/                # Original CSVs (INE, Junta de Andalucía)
│   ├── processed/          # Cleaned datasets (Cities, Hospitals)
│   └── matrix/             # Pre-computed Distance/Time matrices
├── notebooks/              # Data Engineering Pipeline
│   ├── create_cities.ipynb # Cleans municipality data
│   ├── create_hospitals.ipynb # Filters and cleans hospital data
│   ├── create_matrix.ipynb # Calculates Demand/Capacity and Distance Matrices
│   └── create_maps.ipynb   # Generates static HTML maps
├── runs/                   # Output folder for experiments and logs
├── scr/                    # Core Source Code
│   ├── algorithms.py       # Implementation of GA, ACME, Simulated Annealing
│   ├── functions.py        # Fitness functions, Greedy assignment logic
│   ├── experimentacion.py  # Script for batch benchmarking of algorithms
│   ├── multiobjetivo.py    # Multi-objective (Minimize p & Time) logic
│   └── main_multiobjetivo.py # Runner for multi-objective optimization
└── utils/                  # Shared utility functions
```

---

## 🔄 Data Pipeline

The project includes a set of Jupyter Notebooks in `notebooks/` that transform raw data into the format required by the algorithms:

1.  **`create_cities.ipynb`**: Processes `MUNICIPIOS.csv`, filtering for Andalusian provinces and cleaning coordinate data.
2.  **`create_hospitals.ipynb`**: Extracts hospital data from the official catalog (`20251107_Catalogo_de_Hospitales.csv`), filtering for public hospitals in Andalusia.
3.  **`create_matrix.ipynb`**:
    *   Calculates **Hospital Capacity** ($C_j$) based on beds, Length of Stay (LOS), and Occupancy rate.
    *   Calculates **City Demand** ($q_i$) based on population and prevalence rate ($\rho$).
    *   Computes the **Time Matrix** between all cities and hospitals.

---

## 🧠 Algorithms & Logic

The core logic resides in `scr/algorithms.py`. We implement several approaches:

### 1. ACME (A Cooperative Memetic Evolutionary Algorithm)
Our flagship algorithm. It is a hybrid metaheuristic:
*   **Global Search**: A Genetic Algorithm (GA) evolves a population of solutions (sets of hospitals).
*   **Clustering (Memetic)**: Every $k$ generations, the population is clustered using **K-Means**. This identifies distinct "regions" of the search space.
*   **Local Search**: The best individual (medoid) of each cluster undergoes a local search (swapping hospitals) to find the local optimum.
*   **Cooperation**: These optimized individuals are injected back into the population, guiding the evolution.

### 2. Standard Genetic Algorithm (`eaSimple`)
A classic GA used as a baseline for comparison.

### 3. Simulated Annealing
Used both as a standalone solver and as a final refinement step in ACME to polish the best solution found.

### 4. Multi-Objective Approach (`scr/multiobjetivo.py`)
An experimental module that attempts to minimize **two objectives** simultaneously:
1.  Maximum Travel Time ($Z_{max}$)
2.  Number of Open Hospitals ($p$)

---

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/MrCordobex/P-Center-Approach-to-Facility-Location.git
    cd P-Center-Approach-to-Facility-Location
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🖥️ Usage

### Interactive Dashboard
The main interface for decision makers. It allows visualizing the network, changing parameters ($p$, $\rho$), and running the optimization in real-time.

```bash
streamlit run app/app.py
```

### Algorithm Demo
A standalone script to visualize how ACME works on a mathematical function (Griewank). Useful for educational purposes.

```bash
cd ACME
python acme_demo.py
```
*Output: `acme_demonstration.gif`*

### Running Experiments
To compare the performance of different algorithms (ACME vs GA vs SA) across various scenarios:

```bash
python -m scr.experimentacion
```
This will run the cases defined in `scr/experimentacion.py` and save the results to `runs/experimentos_algoritmos.xlsx`.

---

## 👥 Authors

*   **Pedro Martínez Huertas**
*   **Javier Cerón Contreras**

---
*Master in Artificial Intelligence - Universidad Loyola Andalucía*
