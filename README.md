# Revised Proposal to Google Summer of Code 2025  
**[PAL Robotics: Brax Training Viewer for Real-Time Policy Visualization](https://pal-robotics.com/2025-google-summer-code-proposals/)**\
Mujoco.viewer
![Mujoco.viewer demo](https://github.com/bsyh/brax/blob/main/doc/random.gif?raw=true)
Web UI
![Web UI demo](https://github.com/bsyh/brax/blob/main/doc/web_demo.gif?raw=true)

## Application Info

### Cover Letter
Dear PAL Robotics Team,

I’m Bruce, a full-time software developer in Canada and an incoming MS robotics student. I’ve worked as a robotics research assistant (ROS) and a robotics engineer intern (robotic arm).

During the early stages of reinforcement learning (RL) development, researchers often face the repetitive and time-consuming task of tuning hyper-parameters and analyzing training logs. This process can be inefficient, especially without real-time insights into the training progress. As such, I propose the development of a real-time policy visualization tool for Brax, which will allow researchers to observe and debug RL policies as they evolve during training. This tool will significantly reduce the time and effort required for policy development, enabling faster iteration cycles and more effective understanding of the policy black box.

**I have ample commitment time in the summer and wish to maintain my open-source project long-term!** I would like to build a tool useful for my Master's robotics study starting in September. Making it open-source is a great idea, too!

### Contact Details
- **Full name**: Bruce (Shouyue) Hu  
- **Email**: hu.shouyue@outlook.com  
- **Country**: Canada  
- **Resume / CV**: [Link to Resume](doc/resume_bruce_hu.pdf)  
- **Previous contributions links**: [Real-time Blink Detection ROS Package](https://github.com/bsyh/blink_detect_live)  
- **LinkedIn profile**: https://www.linkedin.com/in/brucesh/

## Project Ideas Proposal

### Project Overview
The project focuses on implementing visualization in the MuJoCo XLA (MJX) pipeline using `mujoco.viewer` to create a real-time policy visualization tool for Brax. With the remaining time, a web version will be developed using MuJoCo-WASM, which already displays and visualizes models and environments in 3D dynamics. The web version will connect Brax training data to the front end for real-time visualization.


| **Hours** | **Task**                                                                 |
|-----------|--------------------------------------------------------------------------|
| 24        | Familiarize with source code of JAX/Brax/MuJoCo                          |
| 12        | Investigate UI solutions and write design doc                           |
| 30        | Customize `mujoco.viewer` for policy UI: Add overlays (reward/epoch), real-time plotting |
| 20        | Develop basic workflow with dummy policy training                       |
| 44        | Implement synchronized MuJoCo visualization with Brax training          |
| 20        | Unit tests and validation: Ensure policy actions match visuals          |
| 50        | Support for parallel environments: Visualize multiple agents, enable agent selection |
| 20        | Implement rendering toggle: Pause visualization when disabled           |
| 30        | Large-scale testing: Unit tests, validation, and performance comparison |
| 5        | Refine UI: Polish visuals and improve responsiveness                    |
| 20        | Documentation: User guide with examples, API docs, installation steps   |
| 55        | Develop web version using MuJoCo-WASM: Connect training data to front end for real-time visualization |
| **350**   | **Total**                                                               |

#### Why main data stores in GPU?
Since JAX and MuJoCo XLA (MJX) are designed for accelerators. The tool will peridically pull data from GPU to CPU. 
- Maintain GPU training speed by reducing data trasnfer frequency.
- Let CPU handdle visualization tasks without blocking GPU.

#### Why Web UI?
A web-based UI enables visualization on headless training machines or remote access, providing flexibility for researchers to monitor policies from any device with a browser. Using MuJoCo-WASM simplifies deployment and enhances accessibility.
> Try web UI in [web](https://github.com/bsyh/brax/tree/web) branch

#### Web Version Details
The web version leverages [MuJoCo-WASM](https://github.com/zalo/mujoco_wasm#), which already handles 3D visualization. The 55-hour task involves:
- A communication channel (WebSocket) between the Brax training process and the web server.
- Detailed install doc for community user
- (Optional) A cache mechanism for web.

---

## Contributor's Background
- **Programming Skills**:  
  - 2 years of professional Python experience.  
  - Proficient in MuJoCo simulator, XML, RL training algorithms (PPO), Brax, and JAX.  
  - Experience with scikit-learn, PyTorch, and TensorFlow.  
  - Familiarity with cloud and embedded Linux environments.  

- **Relevant Experience**:  
  - Robotics Engineer Intern  
  - Software Developer Full-time  
  - Robotics Research Assistant  
  - Reinforcement Learning course project (from scratch) using OpenAI/Gym  

## GSoC Participation Details
- **Have you ever participated in a GSoC project?** No  
- **Have you submitted proposals to other organizations for GSoC 2025?** Not yet  

## Code Doc
### File List
```
├── assets/ant.xml
├── assets/ant_policy_params	# Policy learned from train.py         
├── config.py                   # Configuration settings for env parameters
├── train.py                    # Script for training the Ant robot with reinforcement learning
├── viewer_policy.py            # Script for visualizing the learned policy
├── viewer_random.py         	# Script for random control
├── README.md               	# Project documentation
```
