# Neuromorphic Integrated Control Framework 🧠🔄🛰️

## Abstract 📋

Integrated estimation and control present an ongoing challenge for robotic systems. Since controllers rely on data extracted from measured states/parameters which are affected by inherent uncertainties and noises. The suitability of frameworks depends on the complexity of the task and the constraints of computational resources. They must strike a balance between computational efficiency for rapid responses while maintaining accuracy and robustness for safe and reliable missions. This study capitalizes on recent advancements in neuromorphic computing tools, especially spiking neural networks (SNNs), and their applications in robotic and dynamical systems. We present a learning-free framework featuring a recurrent network of leaky integrate-and-fire (LIF) neurons, designed to mimic a linear quadratic regulator (LQR) provided by a robust filtering strategy called extended modified sliding innovation filter (EMSIF). Thus, our proposed framework benefits from the robustness of EMSIF and the computational efficiency of SNN. The weight matrices of SNN are tailored to match the desired system model, eliminating the need for training. Moreover, the network leverages a biologically plausible firing rule akin to predictive coding. Furthermore, in the presence of various uncertainties, the SNN-LQR-EMSIF compared with non-spiking LQR-EMSIF, and the optimal strategy called linear quadratic Gaussian (LQG) based on extended Kalman filter. We evaluate their performance in a workbench problem and, next in the satellite rendezvous maneuver implement the Clohessy-Wiltshire (CW) model. Results demonstrated that the SNN-LQR-EMSIF achieves acceptable performance in terms of computational efficiency, robustness, and accuracy, positioning it as a promising approach for addressing the challenges of Integrated estimation and control in dynamic systems.

**Keywords**: Neuromorphic Computing, Spiking Neural Network, Sliding Innovation Filter, Linear Quadratic Gaussian, Satellite Rendezvous Maneuver, Kalman filter.

## Repository Structure 📁

The repository contains two main case studies:

```
.
├── CASE STUDY 1/            # Basic workbench problem
│   ├── SNN_LQR_SIF.m        # Main script file
│   ├── Cov_dyn.m            # Covariance dynamics function
│   ├── Dyn.m                # System dynamics function
│   ├── KF_Dyn.m             # Kalman Filter dynamics
│   ├── SIF_Dyn.m            # SIF dynamics
│   ├── ...                  # Other utility functions
│
├── CASE STUDY 2/            # Satellite rendezvous maneuver
│   ├── Satellite_Rendezvous_Esti_Cnt.m   # Main script file
│   ├── Cov_dyn.m            # Covariance dynamics function
│   ├── Dyn.m                # System dynamics function
│   ├── KF_Dyn.m             # Kalman Filter dynamics
│   ├── SIF_Dyn.m            # SIF dynamics
│   ├── ...                  # Other utility functions
│
└── README.md                # This documentation file
```

## Prerequisites 🔧

- MATLAB (Tested on R2021b or later)
- No additional toolboxes are required

## Installation 💻

1. Clone this repository:
```bash
git clone https://github.com/yourusername/neuromorphic-integrated-control.git
cd neuromorphic-integrated-control
```

2. Open MATLAB and navigate to the repository directory.

## Running the Case Studies 🏃‍♀️

### Case Study 1: Basic Workbench Problem

This case study demonstrates the effectiveness of the proposed approach on a simpler system.

1. Open MATLAB
2. Navigate to the `CASE STUDY 1` directory
3. Run the main script:
```matlab
run SNN_LQR_SIF.m
```

The script will:
- Initialize system parameters and network settings
- Run simulations comparing LQG, LQR-SIF, and SNN-LQR-SIF approaches
- Generate plots showing:
  - Controlled states for different methods
  - Estimation error with 3σ bounds
  - Spiking pattern visualization for the neural network

#### Modifiable Parameters:

- `NON` (line 40): Network size (number of neurons)
- `landa_SIF` and `landa_KF` (lines 42-43): Time constants for neural dynamics
- `eta_sc_SIF` and `eta_sc_KF` (lines 44-45): Scaling parameters for noise
- `miu` and `nou` (lines 47-48): Network configuration parameters
- `dt` and `tf` (lines 64-65): Simulation time step and final time

### Case Study 2: Satellite Rendezvous Maneuver

This case study applies the framework to a more complex real-world problem using the Clohessy-Wiltshire (CW) model.

1. Open MATLAB
2. Navigate to the `CASE STUDY 2` directory
3. Run the main script:
```matlab
run Satellite_Rendezvous_Esti_Cnt.m
```

The script will:
- Initialize satellite dynamics and orbital parameters
- Configure the control and estimation frameworks
- Run simulations comparing different control approaches
- Generate plots showing:
  - Satellite position and velocity states
  - Estimation errors with 3σ bounds
  - Control input profiles
  - Neural network spiking patterns
  - Percentage of active neurons over time

#### Modifiable Parameters:

- `NON` (line 59): Network size (number of neurons)
- `landa_SIF` and `landa_KF` (lines 61-62): Time constants 
- `eta_sc_SIF` and `eta_sc_KF` (lines 63-64): Scaling parameters
- `miu` and `nou` (lines 66-67): Network configuration parameters
- `dt` and `tf` (lines 85-86): Simulation time step and final time
- Initial conditions `r0` and `v0` (lines 31-32): Initial position and velocity

## Results and Interpretation 📈

The simulation results will show:

1. **State Trajectories**: Compare how each method controls the system states
2. **Estimation Errors**: Visualize how well each method estimates the true states
3. **Spiking Patterns**: Observe the activity of neurons in the SNN
4. **Control Inputs**: Compare the control signals generated by each method

The SNN-LQR-EMSIF approach typically demonstrates:
- Comparable control performance to traditional methods
- Improved robustness against uncertainties
- Computational efficiency through sparse neural activity

## Citation 📚

If you use this code in your research, please cite:

```bibtex
@article{Ahmadvand2025,
  title={Neuromorphic Robust Framework for Integrated Estimation and Control in Dynamical Systems using Spiking Neural Networks},
  author={Ahmadvand, R. and Sharif, S. S. and Banad, Y. M.},
  year={2025}
}
```

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
