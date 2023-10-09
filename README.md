Change to your own fork's badge:

[![MAT-MEK4270 mandatory 1](https://github.com/JouvalSomer/matmek4270-mandatory1/actions/workflows/main.yml/badge.svg)](https://github.com/JouvalSomer/matmek4270-mandatory1/actions/workflows/main.yml)

# 2D PDE Solvers

## Introduction
This repository contains two Python scripts for solving partial differential equations (PDEs) in two dimensions:
- `poisson2d.py`: Solves the 2D Poisson equation.
- `Wave2d.py`: Solves the 2D wave equation.

## Features
- Both solvers employ finite difference methods.
- The Poisson solver utilizes Dirichlet boundary conditions.
- The wave equation solver incorporates both Dirichlet and Neumann boundary conditions.
- Functions for computing L2 error norms and convergence rates are included in both scripts.
- Visualization capabilities using Matplotlib.

## Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib
- SymPy

## Usage

### Poisson 2D Solver
Run the script to solve the 2D Poisson equation and visualize the solution:
```bash
python poisson2d.py
```

### Wave 2D Solver
Run the script to solve the 2D wave equation and generate an animation:
```bash
python Wave2d.py
```
### Testing
Test functions to validate the implementation are included in both scripts. These tests cover:

- Convergence rates.
- Interpolation accuracy (for the Poisson solver).

