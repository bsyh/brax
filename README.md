
Proposal to Google Summer of Code 2025\
[PAL Robotics: Brax Training Viewer for Real-Time Policy Visualization](https://pal-robotics.com/2025-google-summer-code-proposals/)
![demo](doc/random.gif.gif)

# Application Info
### Cover Letter
Dear PAL Robotics Team, 

I’m Bruce, a full-time software developer in Canada and an in-coming MS robotics student. I’ve worked as a robotics research assistant (ROS) and a robotics engineer intern (robotic arm).

During the early stages of reinforcement learning (RL) development, researchers often face the repetitive and time-consuming task of tuning hyper-parameters and analyzing training logs. This process can be inefficient, especially without real-time insights into the training progress. As such, I propose the development of a real-time policy visualization tool for Brax, which will allow researchers to observe and debug RL policies as they evolve during training. This tool will significantly reduce the time and effort required for policy development, enabling faster iteration cycles and more effective understanding in the policy black box.

**I have ample commitment time in the summer and wish to maintain my open-source project in a long-term!** I  would like to build a tool useful for my Master's robotics study starting in September. Making it open-source is a good idea, too!

### Contact Details
- **Full name**: Bruce (Shouyue) Hu
 - **Email**: hu.shouyue@outlook.com 
 - **Country**: Canada 
 - **Resume / CV**: [Link to Resume](doc/resume_bruce_hu.pdf) 
 - **Previous contributions links**: [Real-time Blink Detection ROS Package](https://github.com/bsyh/blink_detect_live) 
 - **LinkedIn profile**: https://www.linkedin.com/in/brucesh/
### Project ideas proposal
Visulization is proposed to be implemented in MuJoCo XLA (MJX) pipeline in HTML format. Such option provides portability to difference platforms.  Considering brax and mujoco is widely used across platforms, portability is a prioritized requiremnt for this project. The use case of brax includs but not limited to: headless embedded linux, cloud TPU/GPU, local AMD/Nvidia GPU or Applle Silicon.

Another option is Qt for cross-platform capability. This option provides better flexability on display content and media types at the cost of an extra dependency on Qt lib.
| **Hours** | **Task** | 
|-----------|--------------------------------------------------------------------------|
| 24 | Familiarize with source code of JAX/Brax/Mojoco |
| 12 | Investigate UI solutions and write design doc |
| 30 | Customize `mujoco.viewer` for policy UI: Add overlays (reward/epoch), support agent selection, real-time plotting | 
| 20 | Develop basic workflow with dummy policy training | 
| 44 | Implement synchronized MuJoCo visualization with Brax training: Extract actions from the policy network at each training step and apply them to a parallel MuJoCo simulation for real-time policy visualization (temperary policy para cache in memory or disk) |
| 20 | Unit tests and validation: Ensure policy actions match visuals |
| 50 | Support for parallel environments: Visualize multiple agents, enable agent selection |
| 20 | Implement rendering toggle: Pause GPU-to-CPU transfer when disabled | 
| 30 | Large-scale testing: Unit tests, validation, and performance comparison |
| 20 | Refine UI: Polish visuals and improve responsiveness |
| 20 | Documentation: User guide with examples, API docs, installation steps |
| 40 | *(Optional)* Optimize GPU-to-CPU data transfer for CUDA |
| 40 | *(Optional)* Optimize GPU-to-CPU data transfer for Apple Silicon |
| **350** | **Total** |


> **Note**: The optional optimization tasks are stretch goals aimed at enhancing performance across different hardware platforms.
> reference: [MLX - A machine learning framework for Apple silicon](https://github.com/ml-explore/mlx)

---

### Technical and Programming Background
- **Programming Skills**:  
  - 2 years of professional Python experience.  
  - Proficient in MuJoCo simulator, XML, RL training algorithms (PPO), Brax, and JAX.  
  - Experience with scikit-learn, PyTorch, and TensorFlow.  
  - Familiarity with cloud and embedded Linux environments.  

- **Relevant Experience**:  
  - Robotics Engineer Intern
  - Software Developer 
  - Robotics Research Assistant
  - Reinforcement Learning course project using OpenAI/Gym

### GSoC Participation details

-   Have you ever participated in a GSoC project? **No**
-   Have you submitted proposals to other organizations for GSoC 2025? **Not yet**

# Code Doc
### File List
```
├── assets/ant.xml
├── assets/ant_policy_params	# Policy learned from train.py         
├── config.py                   # Configuration settings for env parameters
├── train.py                    # Script for training the Ant robot with reinforcement learning
├── viewer_policy.py            # Script for visualizing the learned policy
├── viwer_random.py         	# Script for random control
├── README.md               	# Project documentation
```
