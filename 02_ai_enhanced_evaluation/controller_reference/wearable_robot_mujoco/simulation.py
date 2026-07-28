import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import imageio
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from elbow_control import ElbowMuscleBrain
from ament_index_python.packages import get_package_share_directory

import rclpy
from std_msgs.msg import Float64, Float64MultiArray

from datetime import datetime
from pathlib import Path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#  CSV_PATH = os.path.join("wearable_robot_eval", f"elbow_joint_quaternion_{timestamp}.csv")
BASE_DIR = Path.cwd()
INDEX_FILE = BASE_DIR / "Index.dat"
PATIENT_DIR = BASE_DIR / "Patient"
#CSV_PATH = PATIENT_DIR / f"elbow_joint_quaternion_{timestamp}.csv"

#import xml.etree.ElementTree as ET

def read_patient_name_from_index():
    try:
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            patient_name = f.readline().strip()
        return patient_name
    except Exception as e:
        raise Exception(f"Index.dat Cannot found: {str(e)}")

PATIENT_NAME = read_patient_name_from_index()
CSV_PATH = PATIENT_DIR / f"{PATIENT_NAME}" / "result.csv"

class SimulationReporter:
    def __init__(self, filename):
        self.filename = filename
        self.data = []

    def log(self, info):
        self.data.append(info)

    def save(self):
        import csv
        headers = [
            "timestamp_sec",  # simulation time
            "step",
            "elbow_angle_rad",
            "elbow_qx", "elbow_qy", "elbow_qz", "elbow_qw",
            "target_angle_rad",
            "angular_velocity",
            "target_torque",
            "human_torque",  # modified_1210
            "motor_torque",  # 추가: 모터 토크
            "inter_force" # 추가: 상호작용힘  new_1210
        ]

        with open(self.filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            for row in self.data:
                csv_writer.writerow(row)

        with open(INDEX_FILE, 'w', encoding='utf-8') as f:
            f.write(f"{PATIENT_NAME}\n1\n1\n")

class SimulationNode(Node):
    def __init__(self):
        super().__init__('simulation_node')

        # Create subscriber for commands
        self.vel_cmd_subscriber = self.create_subscription(
            Float64,
            'vel_cmd',
            self.vel_cmd_callback,
            10
        )

        self.torque_cmd_subscriber = self.create_subscription(
            Float64,
            'torque_cmd',
            self.torque_cmd_callback,
            10
        )

        self.position_cmd_subscriber = self.create_subscription(
            Float64,
            'position_cmd',
            self.position_cmd_callback,
            10
        )

        self.stop_cmd_subscriber = self.create_subscription(
            Float64,
            'stop_cmd',
            self.stop_cmd_callback,
            10
        )

        # Create publisher for status (target angle and current angle)
        self.status_publisher = self.create_publisher(
            Float64MultiArray,
            'status',
            10
        )

        # Get XML path from ROS2 package
        share_dir = get_package_share_directory("wearable_robot_mujoco")
        xml_path_ = os.path.join(share_dir, "myoelbow_1dof6muscles_1dofexo.xml")
        self.env = ElbowMuscleBrain("M1", kp=5.0, kd=1.0, xml_path=xml_path_)

        # Read XML and read user attributes from <body> tag
        #  print(xml_path_)
        #  xml_dir = os.path.dirname(xml_path_)
        #  user_xml_path = os.path.join(xml_dir, "assets", "myoelbow_1dof6muscles_1dofexo_body_revised_2.xml")
        #  print(user_xml_path)
        #  root = ET.parse(user_xml_path).getroot()
        #  print(f"Root tag: {root.tag}")
        #  body_ = root.find("worldbody").find("body").find("body").find("body")
        #  print(f"Body tag: {body_.tag}, name: {body_.attrib.get('name')}, user: {body_.attrib.get('user')}")
        #  user_attrib = body_.attrib.get('user').split(' ')
        #  print(f"User attribute: {user_attrib}")

        # Start MuJoCo Viewer
        self.get_logger().info("MuJoCo Viewer Starts...")
        self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data, show_left_ui=False, show_right_ui=False)
        self.renderer = mujoco.Renderer(self.env.model, height=480, width=640)
        self.frames = []

        # Camera position/rotation
        self.viewer.cam.azimuth = 0  
        self.viewer.cam.elevation = 0  
        self.viewer.cam.distance = 2 
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 1.0

        self.cam = mujoco.MjvCamera()
        self.cam.azimuth = 0  
        self.cam.elevation = -20   
        self.cam.distance = 2       
        self.cam.lookat[:] = np.array([0.0, -0.5, 1.4])

        #self.MAX_STEPS = 400 * 0.25
        self.MAX_STEPS = 400 * 10
        self.step_count = 0
        self.infos = []

        # Create timer for simulation loop
        self.timer = self.create_timer(0.01, self.simulation_step)

        # logger for results
        self.report = SimulationReporter(CSV_PATH)

        self.vel_cmd = 0.0
        self.stop_received = False

    def vel_cmd_callback(self, msg):
        self.get_logger().info(f"Received vel_cmd: {msg.data}")
        self.vel_cmd = msg.data
        self.env.set_velocity_command(self.vel_cmd)

    def torque_cmd_callback(self, msg):
        self.get_logger().info(f"Received torque_cmd: {msg.data}")
        torque_cmd = msg.data
        # TODO: set torque command in the environment

    def position_cmd_callback(self, msg):
        self.get_logger().info(f"Received position_cmd: {msg.data}")
        position_cmd = msg.data
        # TODO: set position command in the environment

    def stop_cmd_callback(self, msg):
        self.get_logger().info(f"Received stop_cmd: {msg.data}")
        self.stop_received = True

    def simulation_step(self):
        if not self.viewer.is_running() or self.step_count >= self.MAX_STEPS or self.stop_received:
            self.cleanup()
            self.get_logger().info("Simulation finished.")
            # exit the node
            self.destroy_node()
            sys.exit(0)
            return

        if self.vel_cmd == 0:
            return

        # Env step process
        info = self.env.step()
        self.infos.append(info)

        self.renderer.update_scene(self.env.data, camera=self.cam)
        pixels = self.renderer.render()
        self.frames.append(np.copy(pixels))

        self.step_count += 1

        # MuJoCo Viewer update
        self.viewer.sync()

        # Publish target angle and current angle
        self.publish_status(info)

        # Log data for results.csv
        #sim_time = float(self.env.data.time)
        sim_time = self.env.simulation_time()  # new_1210
        elbow_angle = self.env.data.qpos[8]
        elbow_velocity = self.env.data.qvel[0]

        half_angle = elbow_angle / 2.0
        qx, qy, qz = 0.0, np.sin(half_angle), 0.0
        qw = np.cos(half_angle)

        target_angle = info.get("target_angle", 0.0)
        target_torque = info.get("input", 0.0)
        actual_torque = info.get("actuation", 0.0)

        # 모터 토크 계산 (엑소스켈레톤 모터의 실제 토크)
        motor_torque = (self.env.data.ctrl[6] + self.env.data.ctrl[7]) * self.env.gear  # 모터 토크 명령 × 기어비
        inter_force = self.env.sensor_force()

        row_data = [
            sim_time,
            self.step_count,
            elbow_angle,
            qx, qy, qz, qw,
            target_angle,
            elbow_velocity,
            target_torque,
            actual_torque, # human torque
            motor_torque,  # 추가: 모터 토크
            inter_force # new_1210
        ]
        self.report.log(row_data)

        #self.get_logger().info(f"Step {self.step_count}: ")
        #self.get_logger().info(sim_time.__str__())
        #self.get_logger().info(elbow_angle.__str__())
        #self.get_logger().info(target_angle.__str__())


    def publish_status(self, info): # modified_1208
        simulation_time = self.env.simulation_time()  # new_1210
        target_angle = info["target_angle"]
        sensor_position = self.env.sensor_position()  # position
        sensor_velocity = self.env.sensor_velocity()  # velocity
        sensor_force = self.env.sensor_force()  # muscle force
        msg = Float64MultiArray()
        msg.data = [simulation_time, target_angle, sensor_position, sensor_velocity, sensor_force]
        self.status_publisher.publish(msg)

    def cleanup(self):
        self.get_logger().info(f"Generated result csv: {CSV_PATH}")
        self.report.save()
        self.env.close()
        self.renderer.close()
        self.get_logger().info("MuJoCo Viewer ended.")
        # exit the program


def main(args=None):
    rclpy.init(args=args)
    
    simulation_node = SimulationNode()
    
    try:
        rclpy.spin(simulation_node)
    except KeyboardInterrupt:
        simulation_node.get_logger().info('Simulation interrupted by user')
    finally:
        simulation_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

# # Save mp4
# imageio.mimsave("simulation.mp4", frames, fps=30)
# print("Video saved: simulation.mp4")

# # --- Drawing plots ---
# time_steps = list(range(len(infos)))

# # Extraction
# target_angles  = [info["target_angle"] for info in infos]
# current_angles = [info["current_angle"] for info in infos]
# torques        = [info["input"]         for info in infos]
# actuations     = [info["actuation"]     for info in infos]

# TRIlong_vals   = [info["TRIlong"]       for info in infos]
# TRIlat_vals    = [info["TRIlat"]        for info in infos]
# TRImed_vals    = [info["TRImed"]        for info in infos]
# BIClong_vals   = [info["BIClong"]       for info in infos]
# BICshort_vals  = [info["BICshort"]      for info in infos]
# BRA_vals       = [info["BRA"]           for info in infos]

# plt.figure(figsize=(12, 10))

# # 1. Target Angle vs. Current Angle
# plt.subplot(4, 1, 1)
# plt.plot(time_steps, target_angles, label="Target Angle", color='red')
# plt.plot(time_steps, current_angles, label="Current Angle", color='blue')
# plt.xlabel("Time Step")
# plt.ylabel("Angle (deg)")
# plt.title("Target vs Current Angle")
# plt.legend()
# plt.grid()

# # 2. Torque vs Actuation
# plt.subplot(4, 1, 2)
# plt.plot(time_steps, torques, label="Target Torque", color='black')
# plt.plot(time_steps, actuations, label="Actual Actuation", color='green')
# plt.xlabel("Time Step")
# plt.ylabel("Torque")
# plt.title("Torque Over Time")
# plt.legend()
# plt.grid()

# # 3. Triceps Activations
# plt.subplot(4, 1, 3)
# plt.plot(time_steps, TRIlong_vals, label="TRIlong", color='orange')
# plt.plot(time_steps, TRIlat_vals, label="TRIlat", color='tomato')
# plt.plot(time_steps, TRImed_vals, label="TRImed", color='gold')
# plt.xlabel("Time Step")
# plt.ylabel("Activation")
# plt.title("Triceps Muscle Activations")
# plt.ylim(0, 1)  # Y축 고정
# plt.legend()
# plt.grid()

# # 4. Biceps Activations
# plt.subplot(4, 1, 4)
# plt.plot(time_steps, BIClong_vals, label="BIClong", color='cyan')
# plt.plot(time_steps, BICshort_vals, label="BICshort", color='dodgerblue')
# plt.plot(time_steps, BRA_vals, label="BRA", color='blueviolet')
# plt.xlabel("Time Step")
# plt.ylabel("Activation")
# plt.title("Biceps Muscle Activations")
# plt.ylim(0, 1)  # Y축 고정
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()
