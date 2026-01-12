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

### 1. Clone the repository
```bash
git clone https://github.com/yourname/yourrepo.git
cd yourrepo

mpc:
  horizon: 20
  dt: 0.1
  max_velocity: 1.0
  safety_radius: 0.4
  Q_pos: 5.0
  Q_theta: 1.0
  R_v: 0.2
  R_w: 0.1

planner:
  type: RRT
  max_nodes: 1000

experiment:
  runs_per_goal: 10
  number_of_goals: 10
  max_time: 60



---

Als je wilt kan ik hierna ook:
- een **GitHub-ready badge header** toevoegen  
- of een **academisch paper-style README** maken  
- of een **bedrijfsvriendelijke versie voor Qafka** 💼

