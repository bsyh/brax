# config.py

# Number of environments to simulate
NUM_ENVS = 4  

# Distance between environments in MuJoCo world space
ENV_SEPARATION = 3.0

# Number of environments per row in the grid layout
ENVS_PER_ROW = 2  

# Path to the MuJoCo model XML file
MODEL_PATH = "assets/ant.xml"  # Update with the correct path

# Duration of the simulation in seconds
DURATION = 10  

# Frame rate for the viewer (Hz)
FRAMERATE = 60  

# Control range for actuators
CONTROL_RANGE = (-1.0, 1.0)  
