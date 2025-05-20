# Generative reinforcement learning for fluid antenna systems design

### This is the codebase for our paper "Generative reinforcement learning for fluid antenna systems design".
---

## Overview
The fluid antenna systems design often involve the high-dimension mixed-integer non-convex optimization problems, while the traditional optimization algorithms undergo the challenges of high computational complexity and poor performance. This paper proposes a generative reinforcement learning algorithm, which leverages a primary-dual network (PD-Net)  and the generative adversarial network-enhanced proximal policy optimization (GPPO) algorithm for beamforming optimization and port selection respectively. The proposed algorithm can obtain higher quality solution without the need for training labels. Via three cases, the superiority of the proposed algorithm over the benchmarks is verified. Moreover, the PD-Net can deal with the beamforming optimization problem with complicated equality and inequality constraints, and the GPPO algorithm illustrates better performance and stability than proximal policy optimization (PPO) benchmark, verifying the effectiveness of generative adversarial training. 


![GPPO-enabled fluid antenna systems](./figures/fas.png)

## Experimental platform

### Communication-only Case
To use the provided code, you are supposed to:

1. run ```python simulation/com-only/GAN_PPO/main.py --agent_name GANPPO``` obtain ```GANPPO_la_1e-05_lc_0.0001_hid_128_eps_0.4_gamma_0.9.npz```
2. run ```python simulation/com-only/GAN_PPO/main.py --agent_name PPO``` obtain ```PPO_la_1e-05_lc_0.0001_hid_128_eps_0.4_gamma_0.9.npz``` 
3. run ```python simulation/com-only/GAN_PPO/main_reward_vs_ns.py --agent_name GANPPO``` obtain ```GANPPO_reward_vs_ns.npy```
4. run ```python simulation/com-only/GAN_PPO/main_reward_vs_ns.py --agent_name PPO``` obtain ```PPO_reward_vs_ns.npy```
5. run ```python simulation/com-only/compare_distribution.py```, ```simulation/com-only/compare_performance.py``` to obtain the simulation results in Fig. 2 of our manuscript.

### Sensing-only Case
To use the provided code, you are supposed to:

1. run ```python simulation/sensing-only/GAN_PPO/main.py --agent_name GANPPO``` obtain ```GANPPO_la_1e-05_lc_0.0001_hid_128_eps_0.4_gamma_0.8.npz```
2. run ```python simulation/sensing-only/GAN_PPO/main.py --agent_name PPO``` obtain ```PPO_la_1e-05_lc_0.0001_hid_128_eps_0.4_gamma_0.8.npz``` 
3. run ```python simulation/sensing-only/GAN_PPO/main_reward_vs_ns.py --agent_name GANPPO``` obtain ```GANPPO_reward_vs_ns.npy```
4. run ```python simulation/sensing-only/GAN_PPO/main_reward_vs_ns.py --agent_name PPO``` obtain ```PPO_reward_vs_ns.npy```
5. run ```python simulation/sensing-only/compare_distribution.py```, ```python simulation/sensing-only/compare_performance.py``` to obtain the simulation results in Fig. 3 of our manuscript.
6. run ```python simulation/sensing-only/compare_beampattern.py``` to obtain the simulation results in Fig. 4 in our manuscript.

### ISAC Case
To use the provided code, you are supposed to:

1. run ```python simulation/isac/GAN_PPO/main.py --agent_name GANPPO``` obtain ```GANPPO_la_1e-05_lc_8e-05_hid_128_eps_0.4_gamma_0.8.npz```
2. run ```python simulation/isac/GAN_PPO/main.py --agent_name PPO``` obtain ```PPO_la_1e-05_lc_8e-05_hid_128_eps_0.4_gamma_0.8.npz``` 
3. run ```python simulation/isac/GAN_PPO/main_reward_vs_ns.py --agent_name GANPPO``` obtain ```GANPPO_reward_vs_ns.npy```
4. run ```python simulation/isac/GAN_PPO/main_reward_vs_ns.py --agent_name PPO``` obtain ```PPO_reward_vs_ns.npy```
5. run ```python simulation/isac/compare_distribution.py```, ```python simulation/isac/compare_performance.py``` to obtain the simulation results in Fig. 5 in our manuscript.


## Experimental results
![GPPO-enabled fluid antenna systems](./figures/compare_com_reward_vs_ns.png)

## Requirement
All the simulation results were conducted on a Dell Precision 7960 server (System: Ubuntu 20.04, CPU: Intel(R) Xeon(R) w7-3455, CPU clock: 2.5 GHz, RAM: LPDDR4 128GB, GPU: NVIDIA RTX 4000Ada 20GB) The proposed algorithms are implemented in Python 3.8.20 with Pytorch 2.4.1. 

1. Python 3.8+
2. Pytorch 2.4.1
3. NVIDIA RTX 4000Ada 20GB
4. Ubuntu 20.04
5. install the software packages via ```pip install -r requirement.txt```