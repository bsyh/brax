"""Multi-ant simulation in MuJoCo using a Brax-trained PPO policy."""

import mujoco
import time
import numpy as np
from mujoco import viewer
import xml.etree.ElementTree as ET
import jax
import jax.numpy as jnp
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo

import config


def load_policy():
    """Loads a pre-trained PPO policy for the Brax ant environment.

    Returns:
        tuple: (brax_env, jit_inference_fn) where brax_env is the environment
               and jit_inference_fn is the JIT-compiled policy inference function.
    """
    env_name = "ant"
    backend = "positional"
    
    # Create the Brax environment
    env = envs.create(env_name=env_name, backend=backend)
    
    # Load trained policy parameters
    params = model.load_params("policy/ant_policy_params")
    
    # Create inference function using the same PPO training setup
    make_inference_fn, _, _ = ppo.train(
        environment=env, num_timesteps=0, episode_length=1
    )
    inference_fn = make_inference_fn(params)
    
    # JIT compile for performance
    jit_inference_fn = jax.jit(inference_fn)
    
    return env, jit_inference_fn


def create_multi_robot_env(num_envs, env_separation, envs_per_row, model_path):
    """Creates a MuJoCo model with multiple ant robots in a grid layout.

    Args:
        num_envs (int): Number of ant robots to create.
        env_separation (float): Distance between robots in the grid (meters).
        envs_per_row (int): Number of robots per row in the grid.
        model_path (str): Path to the base MuJoCo XML file for a single ant.

    Returns:
        tuple: (model, data) where model is the MuJoCo model and data is the
               simulation data object.

    Raises:
        ValueError: If no torso body is found in the XML file.
    """
    # Parse the base XML file
    tree = ET.parse(model_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    
    # Find the base robot torso
    base_robot = worldbody.find("body[@name='torso']")
    if base_robot is None:
        raise ValueError("No robot model found in the XML file.")
    
    # Calculate grid dimensions
    num_rows = (num_envs + envs_per_row - 1) // envs_per_row
    
    # Replicate the robot across the grid
    for i in range(1, num_envs):
        row = i // envs_per_row
        col = i % envs_per_row
        x_pos = col * env_separation
        y_pos = row * env_separation
        
        # Clone the base robot
        new_robot = ET.fromstring(ET.tostring(base_robot))
        new_robot.set("name", f"robot_{i}")
        new_robot.set("pos", f"{x_pos} {y_pos} 0")
        
        # Rename all elements to avoid name conflicts
        for elem in new_robot.iter():
            if "name" in elem.attrib:
                elem.set("name", f"{elem.attrib['name']}_{i}")
        
        worldbody.append(new_robot)

    # Clone actuators for each robot
    actuator_root = root.find("actuator")
    if actuator_root is not None:
        base_actuators = list(actuator_root)
        for i in range(1, num_envs):
            for base_actuator in base_actuators:
                new_actuator = ET.fromstring(ET.tostring(base_actuator))
                if "name" in new_actuator.attrib:
                    new_actuator.set("name", f"{new_actuator.attrib['name']}_{i}")
                if "joint" in new_actuator.attrib:
                    new_actuator.set("joint", f"{new_actuator.attrib['joint']}_{i}")
                actuator_root.append(new_actuator)
    
    # Convert to MuJoCo model
    xml_string = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    return model, data


def get_observation(data, robot_idx):
    """Extracts the observation for a single ant in Brax-compatible format.

    Args:
        data (mujoco.MjData): MuJoCo simulation data object.
        robot_idx (int): Index of the robot to extract observation for.

    Returns:
        jax.numpy.ndarray: Observation vector of shape (27,) containing z-position,
                           orientation, joint angles, and velocities.
    """
    # Ant-specific DOF counts
    qpos_per_robot = 15  # 7 for torso (pos + quat) + 8 joints
    qvel_per_robot = 14  # 6 for torso (linear + angular vel) + 8 joint velocities
    
    # Calculate start indices for this robot's state
    qpos_start = robot_idx * qpos_per_robot
    qvel_start = robot_idx * qvel_per_robot
    
    # Extract position and velocity slices
    qpos = data.qpos[qpos_start:qpos_start + qpos_per_robot]
    qvel = data.qvel[qvel_start:qvel_start + qvel_per_robot]
    
    # Construct Brax-compatible observation (skip x, y position)
    obs = jnp.concatenate([
        qpos[2:],  # z, quaternion, joints (13 elements)
        qvel       # all velocities (14 elements)
    ])
    
    return obs


def main():
    """Runs the multi-ant simulation with a Brax-trained policy."""
    # Load the policy and environment
    _, jit_inference_fn = load_policy()
    
    # Create the multi-robot MuJoCo environment
    model, data = create_multi_robot_env(
        config.NUM_ENVS,
        config.ENV_SEPARATION,
        config.ENVS_PER_ROW,
        config.MODEL_PATH
    )
    
    # Calculate controls per robot
    controls_per_robot = model.nu // config.NUM_ENVS
    
    # Initialize random number generator
    rng = jax.random.PRNGKey(seed=1)
    
    # Launch the MuJoCo viewer
    with viewer.launch_passive(model, data) as v:
        # Reset the simulation
        mujoco.mj_resetData(model, data)
        start_time = time.time()
        
        # Main simulation loop
        while time.time() - start_time < config.DURATION:
            # Update controls for each robot
            for robot_idx in range(config.NUM_ENVS):
                # Get observation for this robot
                obs = get_observation(data, robot_idx)
                
                # Generate action using the policy
                act_rng, rng = jax.random.split(rng)
                action, _ = jit_inference_fn(obs, act_rng)
                
                # Apply action to this robot's control inputs
                ctrl_start = robot_idx * controls_per_robot
                ctrl_end = (robot_idx + 1) * controls_per_robot
                data.ctrl[ctrl_start:ctrl_end] = np.array(action[:controls_per_robot])
            
            # Advance the simulation
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(1 / config.FRAMERATE)
        
        # Keep viewer open until closed manually
        while v.is_running():
            v.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()