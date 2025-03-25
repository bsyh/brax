import mujoco
import time
import numpy as np
from mujoco import viewer
import config
import xml.etree.ElementTree as ET

import os
import pickle
import jax.numpy as jnp

def save_model(model_state, filename):
    """Saves the trained model state to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model_state, f)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Loads a trained model state from a file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found.")
    with open(filename, 'rb') as f:
        model_state = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model_state

def evaluate_policy(env, model_state, num_episodes=10):
    """Evaluates the trained policy on the given environment."""
    total_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = model_state.policy(state.obs)  # Get action from policy
            state = env.step(action)
            episode_reward += state.reward
            done = state.done
        total_rewards.append(episode_reward)
    avg_reward = jnp.mean(jnp.array(total_rewards))
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

def create_multi_robot_env(num_envs, env_separation, envs_per_row, model_path):
    """
    Creates a multi-environment MuJoCo model by replicating a robot across `num_envs` independent spaces.
    
    Args:
        num_envs (int): Number of independent environments.
        env_separation (float): Distance between environments.
        envs_per_row (int): Number of environments per row.
        model_path (str): Path to the MuJoCo XML model file.
    
    Returns:
        mujoco.MjModel: The compiled MuJoCo model.
        mujoco.MjData: Simulation data associated with the model.
    """
    # Load base XML
    tree = ET.parse(model_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    
    # Ensure base robot exists
    base_robot = worldbody.find("body[@name='torso']")  # Locate torso as the base of the robot
    if base_robot is None:
        raise ValueError("No robot model found in the XML file.")
    
    # Calculate grid dimensions
    num_rows = (num_envs + envs_per_row - 1) // envs_per_row
    
    # Replicate robots
    for i in range(1, num_envs):  # Start at 1 since we keep the original
        row = i // envs_per_row
        col = i % envs_per_row
        x = col * env_separation
        y = row * env_separation
        
        new_robot = ET.fromstring(ET.tostring(base_robot))
        new_robot.set('name', f'robot_{i}')
        new_robot.set('pos', f'{x} {y} 0')
        
        # Ensure all child elements (joints, actuators, etc.) have unique names
        for elem in new_robot.iter():
            if 'name' in elem.attrib:
                elem.set('name', f"{elem.attrib['name']}_{i}")
        
        worldbody.append(new_robot)

    # Find the actuators section and clone actuators for each robot
    actuator_root = root.find('actuator')
    if actuator_root is not None:
        base_actuators = list(actuator_root)
        for i in range(1, num_envs):  # Start at 1 to keep original
            for base_actuator in base_actuators:
                new_actuator = ET.fromstring(ET.tostring(base_actuator))
                
                # Update actuator name
                if 'name' in new_actuator.attrib:
                    new_actuator.set('name', f"{new_actuator.attrib['name']}_{i}")
                
                # Update the joint it controls
                if 'joint' in new_actuator.attrib:
                    new_actuator.set('joint', f"{new_actuator.attrib['joint']}_{i}")
                
                actuator_root.append(new_actuator)
    
    # Convert modified XML to string and load into MuJoCo
    xml_string = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    return model, data

def main():
    """Runs the multi-environment simulation."""
    # Create the environment
    m, d = create_multi_robot_env(
        config.NUM_ENVS,
        config.ENV_SEPARATION,
        config.ENVS_PER_ROW,
        config.MODEL_PATH
    )
    
    # Determine the number of controls per robot dynamically
    controls_per_robot = m.nu // config.NUM_ENVS
    total_controls = config.NUM_ENVS * controls_per_robot
    
    with viewer.launch_passive(m, d) as v:
        mujoco.mj_resetData(m, d)
        t0 = time.time()
        
        while time.time() - t0 < config.DURATION:
            # Apply random controls
            d.ctrl[:] = np.random.uniform(*config.CONTROL_RANGE, size=(config.NUM_ENVS, controls_per_robot)).flatten()
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(1/config.FRAMERATE)
        
        while v.is_running():
            v.sync()
            time.sleep(1/60)

if __name__ == "__main__":
    main()
