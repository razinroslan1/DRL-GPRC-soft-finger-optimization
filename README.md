# DRL-GPRC-soft-finger-optimization
This repository contains the Python implementation of a Deep Reinforcement Learning framework for optimizing the design of a soft robotic finger using Finite Element Method (FEM) simulations and Gaussian Process Residual Correction (GPRC).

## Structure
- `main.py`: Main script to run DRL optimization
- `gprc.py`: Gaussian Process Residual Correction 
- `fem.py`: Ansys MAPDL solver
- `environment.py`: Reinforcement Learning environment
- `data/`: Experimental data
- `STL`: STL files of printed soft robotic fingers

## Requirements
- Python 3.10
- Ansys MAPDL
- Dependencies listed in requirements.txt

## Installation
Clone the repository and install the required Python dependencies:
```Bash
pip install -r requirements.txt
```

## Usage
Run the optimization using:
```
python main.py
```

GPRC can be enabled or disabled with the variable use_gprc in main.py:
```
use_gprc = True # GPRC enabled
use_gprc = False # GPRC disabled
```

## Related Publications
If you wish to use, reproduce, or adapt any code or designs from this repository in your own work, please cite the associated research paper, currently under review:

**Residual Learning with Human-Assisted Experimental Feedback for Embodied Sim-to-Real Design Optimization of Soft Robotic Fingers**

Authors: Muhammad Razin Roslan, Koay Jin Yuan, Bryan Jun Sheng Lee, Surya G. Nurzaman

Journal: Biomimetic Intelligence and Robotics

DOI: TBA