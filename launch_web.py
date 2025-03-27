import mujoco
import xml.etree.ElementTree as ET
import config

import http.server
import socketserver
import os

def start_web_server(directory_path, port=8000):
    """
    Starts a simple HTTP web server to serve files from the specified directory.
    
    Args:
        directory_path (str): The path to the directory to serve files from.
        port (int): The port number to run the server on (default is 8000).
    """
    # Change to the specified directory
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory '{directory_path}' does not exist.")
    os.chdir(directory_path)
    
    # Set up the HTTP server
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), Handler)
    
    # Print server details
    print(f"Serving HTTP on http://localhost:{port} from directory: {os.path.abspath(directory_path)}")
    print("Press Ctrl+C to stop the server.")
    
    # Start the server
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()

def create_multi_robot_env(num_envs, env_separation, envs_per_row, model_path, output_path):
    """
    Creates a multi-environment MuJoCo model by replicating a robot across `num_envs` independent spaces
    and saves the modified XML to a file.
    
    Args:
        num_envs (int): Number of independent environments.
        env_separation (float): Distance between environments.
        envs_per_row (int): Number of environments per row.
        model_path (str): Path to the MuJoCo XML model file.
        output_path (str): Path where the modified XML will be saved. Defaults to "multi_robot_env.xml".
    
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
    
    # Save the modified XML to a file
    xml_string = ET.tostring(root, encoding='unicode', method='xml')
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')  # Add XML declaration
        f.write(xml_string)
    
    # Create and return MuJoCo model and data
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    return model, data

if __name__ == "__main__":
    # Create the multi-robot MuJoCo environment
    model, data = create_multi_robot_env(
        config.NUM_ENVS,
        config.ENV_SEPARATION,
        config.ENVS_PER_ROW,
        config.MODEL_PATH,
        "wasm/examples/scenes/multi_ants.xml"
    )


    # Specify the path where your files (e.g., XML) are located
    path_to_serve = "wasm"  # Replace with your actual path
    start_web_server(path_to_serve, port=8000)