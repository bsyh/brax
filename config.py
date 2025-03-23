# config.py
"""Configuration parameters for the multi-ant MuJoCo simulation."""

# Number of ant robots to simulate
NUM_ENVS = 4

# Distance between each ant environment in the grid (meters)
ENV_SEPARATION = 2.0

# Number of ants per row in the grid layout
ENVS_PER_ROW = 2

# Path to the base MuJoCo XML model file for a single ant
MODEL_PATH = "assets/ant.xml"  # Adjust this to your actual ant XML file path

# Duration of the simulation in seconds
DURATION = 10.0

# Frame rate for rendering (frames per second)
FRAMERATE = 60

# Control range for each robot
CONTROL_RANGE = (-1.0, 1.0)