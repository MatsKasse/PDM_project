# RO47005 Planning and Decision Making (2025/26 Q2)

End-to-end motion planning and control framework for a differential-drive mobile robot using Model Predictive Control (MPC) with global path planning and collision avoidance.

---

## 📌 Project Overview

This repository describes motion planning for a mobile service robot operating in a dynamic restaurant environment. Different global path planners are combined with a local Model Predictive Control (MPC) planner, to compare how the global planners perform. As a global planner, Rapidly-exploring Random Trees (RRT) is used as a baseline, and is compared to an RRT* and A* planner. The robot model is a non-holonomic differential-drive robot. 

The system combines:

- **Global planning** (RRT, RRT*, A*)
- **Local trajectory tracking via MPC**
- **Soft-constrained obstacle avoidance**
- **Physics-based simulation (PyBullet)**

The framework is designed for **quantitative benchmarking** of different global planners under identical local control and environment conditions.
The system was developed within the TU Delft MSc Robotics PDM project and follows engineering practices that are compatible with real-world deployment.

---

## 🧠 Core Capabilities

- Linear MPC formulated as a Quadratic Program (QP)
- OSQP as real-time optimization solver
- Soft collision constraints using slack variables
- Static and dynamic obstacle support
- Path tracking from global planners
- Automatic logging and export of metrics

---

## 🗂 Repository Structure

The **entire pipeline** is controlled via `runner_file.py`, which executes global planning, MPC control, simulation, and result logging.

---

## ⚙️ System Requirements

### Software
- Ubuntu 20.04+ (recommended)
- Python 3.9 – 3.11
- Git

### Core Python dependencies
Installed automatically:
- NumPy  
- SciPy  
- OSQP  
- PyBullet  
- Gymnasium  
- Shapely  
- Pandas  
- Matplotlib  

---

## 🚀 Installation & Setup

### 1. Clone the repository and Run
```bash
git clone https://github.com/yourname/yourrepo.git](https://github.com/MatsKasse/PDM_project.git

cd PDM_project

python runner_file.py
```
## ⚙️ Configuration and Tuning

All experiment settings are centrally controlled via `runner_file.py`.  
Within this file you can configure:

- The selected **global planner** (RRT, RRT*, A*) and its **parameters**
- **MPC parameters** (horizon, weights, velocity limits, safety radius)
- **Simulation parameters** (number of runs, renders, dynamic obstacles)

The file contains inline comments explaining each parameter, making it straightforward to modify the behavior of the planners, controller, and simulation with minimal changes in the core of the code.

---

## 👤 Authors
  
- Mats Kasse; m.kasse@student.tudelft.nl
- Bram Kalkman; bakalkman@student.tudelft.nl
- Lars Emmer; lemmer@student.tudelft.nl
- Jesper van der Meulen; jespermeulen@student.tudelft.nl

For questions regarding the implementation, experiments, or results, please contact the authors.

