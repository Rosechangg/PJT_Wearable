#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32
from wearable_robot_upper_limb_msgs.srv import UpperLimbCommand
from wearable_robot_upper_limb_msgs.msg import UpperLimbState
from geometry_msgs.msg import Vector3
import time
from dynamixel_sdk import *
import threading

import math
from .wearable_robot_api import Upperlimb_1DOF

# Control table address
# https://emanual.robotis.com/docs/en/dxl/x/xh430-v350/#control-table-description
ADDR_BAUDRATE           = 8
ADDR_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
ADDR_GOAL_VELOCITY      = 104
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_LOAD       = 126
ADDR_PRESENT_VELOCITY   = 128
ADDR_PRESENT_POSITION   = 132
ADDR_PROFILE_VELOCITY   = 112
ADDR_OPERATING_MODE     = 11
ADDR_MIN_POSITION_LIMIT = 48
ADDR_MAX_POSITION_LIMIT = 52
ADDR_HARDWARE_ERROR_STATUS = 70

OP_CURRENT_BASED_POSITION = 5
OP_VELOCITY = 1

# Protocol version
PROTOCOL_VERSION            = 2.0

# Default setting
DXL_ID                      = 1                 # Dynamixel ID : 1
#  BAUDRATE                    = 57600             # Dynamixel default baudrate : 57600, Dynamixel XH430-V350-R series
BAUDRATE                    = 2000000           # Dynamixel XH540-W270-R series
DEVICENAME                  = '/dev/ttyUSB0'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque

# Protocol V2; XH430-V350-R, XM540-W270-R Range Information
# https://emanual.robotis.com/docs/en/dxl/x/xh430-v350/
# https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/
DXL_MINIMUM_POSITION_VALUE  = -2147483648                 # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE  = 2147483647              # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_MINIMUM_VELOCITY_VALUE  = -1023                 # Minimum velocity value
DXL_MAXIMUM_VELOCITY_VALUE  = 1023              # Maximum velocity value
DXL_MINIMUM_CURRENT_VALUE   = -689                 # Minimum current value
DXL_MAXIMUM_CURRENT_VALUE   = 689              # Maximum current value
DXL_MOVING_STATUS_THRESHOLD = 10                # Dynamixel moving status threshold
MAX_VALUE_4BYTES = 4294967295  # 2^32 - 1
MAX_VALUE_2BYTES = 65535      # 2^16 - 1

UNIT_DEGREE_VALUE = 0.088
UNIT_RPM_VALUE = 0.229
#  UNIT_MILLI_AMPERE_VALUE = 1.34 # XH430-V350-R
UNIT_MILLI_AMPERE_VALUE = 2.69 # XH540-w270-R https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/#goal-current102

class DynamixelController:
    def __init__(self, node_logger):
        self.logger = node_logger
        
        # Constants
        self.DXL_ID = DXL_ID
        self.PROTOCOL_VERSION = PROTOCOL_VERSION
        self.BAUDRATE = BAUDRATE
        self.DEVICENAME = DEVICENAME

        # Initialize Dynamixel
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
        
        self._initialize_connection()

    def _initialize_connection(self):
        if self.portHandler.openPort():
            self.logger.info("Succeeded to open port")
        else:
            self.logger.error("Failed to open port")
            return False
            
        if self.portHandler.setBaudRate(self.BAUDRATE):
            self.logger.info("Succeeded to change baudrate")
        else:
            self.logger.error("Failed to change baudrate")
            return False
        return True

    def ping(self):
        dxl_model_number, dxl_comm_result, dxl_error = self.packetHandler.ping(self.portHandler, self.DXL_ID)
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            self.logger.info("[ID:%03d] ping Succeeded. Dynamixel model number : %d" % (self.DXL_ID, dxl_model_number))

    def disable_torque(self):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            self.logger.info("Disabled torque")

    def enable_torque(self):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            self.logger.info("Enabled torque")
        
    def set_operating_mode(self, mode):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, ADDR_OPERATING_MODE, mode)
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            self.logger.info(f"Set operating mode to {mode}")
       
    def set_baudrate(self, baudrate):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, ADDR_BAUDRATE, baudrate)
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def write_control_table(self, addr=ADDR_PROFILE_VELOCITY, value=30):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL_ID, addr, value)
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))

    # {GET_POSITION, PRESENT_POSITION, UNIT_DEGREE, -2147483647 , 2147483647, 0.088},
    def get_present_position(self):
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID, ADDR_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))

        if dxl_present_position > DXL_MAXIMUM_POSITION_VALUE:
            dxl_present_position = dxl_present_position - MAX_VALUE_4BYTES

        return dxl_present_position * UNIT_DEGREE_VALUE # (360/4096) = 0.088 

    def get_present_velocity(self):
        dxl_present_velocity, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, self.DXL_ID, ADDR_PRESENT_VELOCITY)
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))

        if dxl_present_velocity > DXL_MAXIMUM_VELOCITY_VALUE:
            dxl_present_velocity = dxl_present_velocity - MAX_VALUE_4BYTES

        return dxl_present_velocity * UNIT_RPM_VALUE

    def get_present_current(self):
        current, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(
            self.portHandler, self.DXL_ID, ADDR_PRESENT_LOAD
        )
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))
        
        if current > DXL_MAXIMUM_CURRENT_VALUE:
            current = current - MAX_VALUE_2BYTES

        return current * UNIT_MILLI_AMPERE_VALUE

    # {SET_POSITION, GOAL_POSITION, UNIT_DEGREE, -1048575, 1048575, 0.088},
    def set_goal_position(self, position):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, self.DXL_ID, ADDR_GOAL_POSITION, 
            int(position / UNIT_DEGREE_VALUE)
        )
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def set_goal_velocity(self, velocity):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, self.DXL_ID, ADDR_GOAL_VELOCITY, 
            int(velocity / UNIT_RPM_VALUE)
        )
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))
       
    # TODO: Need to test or remove
    def set_min_position_limit(self, position):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, self.DXL_ID, ADDR_MIN_POSITION_LIMIT, 
            int(position / UNIT_DEGREE_VALUE)
        )
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))

    # TODO: Need to test or remove
    def set_max_position_limit(self, position):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, self.DXL_ID, ADDR_MAX_POSITION_LIMIT,
            int(position / UNIT_DEGREE_VALUE)
        )
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))

    def reboot(self):
        self.logger.info("Reboot")
        dxl_comm_result, dxl_error = self.packetHandler.reboot(self.portHandler, self.DXL_ID)
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))
        time.sleep(3)

    def get_hardware_error_status(self):
        dxl_present_hw_error, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler, self.DXL_ID, ADDR_HARDWARE_ERROR_STATUS)
        if dxl_comm_result != COMM_SUCCESS:
            self.logger.info("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            self.logger.info("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            self.logger.info("[ID:%03d] Get Hardware Error Status Succeeded. Error : %d" % (self.DXL_ID, dxl_present_hw_error))
        return dxl_present_hw_error

    def close(self):
        self.disable_torque()
        self.portHandler.closePort()
        self.logger.info("Port closed")

class UpperLimbNode(Node):
    def __init__(self):
        super().__init__('wearable_robot_upper_limb_controller')
        
        # loop time
        self.DELTA_TIME = 0.0125  # 80Hz
        self.REP_COUNT = 21

        # running status
        self.selected_task = 0
        self.is_running = False
        self.is_sending_upper_limb_state = False
        
        # Parameters
        self.declare_parameter('robot_weight', 200.0)  # g
        self.declare_parameter('l1', 0.18)  # m
        self.declare_parameter('l2', 0.30)  # m
        self.declare_parameter('repeat', self.REP_COUNT) # start from zero, total 20 flexion and extention and extra flexion and extention for finishing
        self.declare_parameter('delay_time', 1000)  # ms
        
        # Control parameters
        self.m = 0.5
        self.c = 10
        self.k = 0
        self.k2 = 1
        self.a = -90/2
        
        # State variables
        self.current_time = 0.0
        self.direction = -1
        self.flag = 2
        self.r = -1
        self.start = 0
        self.theta = 0.0

        # Initialize Dynamixel
        self.previous_velocity = 0.0

        # Upperlimb_1DOF API
        self.sensor_velocity = 0.0
        self.sensor_force = 0.0
        self.sensor_position = 0.0
        self.target_th = 0.0
        self.upperlimb = Upperlimb_1DOF(node=self)  # Gets singleton instance and sets node
        self._load_task_module()

        # Publishers
        self.upper_limb_state_pub = self.create_publisher(UpperLimbState, 'upper_limb_status', 10)
        
        # Subscribers
        self.loadcell_value = None
        self.loadcell_sub = self.create_subscription(Float32, 'load_cell_weight', self.loadcell_callback, 10)

        # Services
        self.command_service = self.create_service(UpperLimbCommand, 'upper_limb_command', self.handle_command)
        
        # Timer for control loop (80Hz)
        self.timer = self.create_timer(self.DELTA_TIME, self.control_loop)

        # Motor controller
        self.dxl = DynamixelController(self.get_logger())

        # Reste hardware errors
        dynamixel_reboot_tries = 0
        while(self.dxl.get_hardware_error_status() != 0 and dynamixel_reboot_tries < 5):
            self.get_logger().info("Try to reset hardware errors of dynamixel motor")
            self.dxl.reboot()
            time.sleep(3)
            dynamixel_reboot_tries += 1

        self.dxl.reboot()
        self.reset_motor_position()


    def __del__(self):
        self.stop()
        self.dxl.close()
        self.get_logger().info("__del__ Port closed")

    def _load_task_module(self):
        """Load task module and call its setup function"""
        try:
            from . import task
            self.task_module = task
            task.setup()
            self.get_logger().info('task module loaded and setup() called successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load task module: {e}')
            self.task_module = None

    def reset_motor_position(self):
        self.dxl.ping() # ping to check if the motor is connected

        self.dxl.disable_torque() # Disable Dynamixel Torque
        self.dxl.set_operating_mode(OP_CURRENT_BASED_POSITION) # Set operating mode to current-based position for repositioning
        self.dxl.enable_torque() # Enable Dynamixel Torque
 
        self.dxl.write_control_table(ADDR_PROFILE_VELOCITY, 30)
        self.dxl.set_goal_position(170.0)

        # reposition motor to 170 degree
        current_position = self.dxl.get_present_position()
        while abs(current_position - 170.0) > DXL_MOVING_STATUS_THRESHOLD: # Wait until the 170 goal position is reached
            current_position = self.dxl.get_present_position()
            time.sleep(0.1)

        time.sleep(1.0)
        self.dxl.write_control_table(ADDR_PROFILE_VELOCITY, 60)
        time.sleep(2.0)
        self.get_logger().info("Motor reposition is complete")

        #self.dxl.disable_torque() # Disable Dynamixel Torque
        #self.dxl.set_operating_mode(OP_VELOCITY) # Set operating mode to velocity
        #self.dxl.enable_torque() # Enable Dynamixel Torque

    #  def start(self):
    #      self.timer.reset()
    #      self.timer.start()
    #      self.dxl.enable_torque()
    #      self.get_logger().info('Starting...')
    
    def stop(self):
        self.timer.cancel()
        self.dxl.disable_torque()
        self.get_logger().info('Stopping...')

    def control_loop(self):
        if not self.is_running:
            self.dxl.set_goal_velocity(0)
            if self.is_sending_upper_limb_state:
                current_position = self.dxl.get_present_position()
                loadcell_value = self.read_loadcell()  # Implement according to your HX711 interface
                current = self.dxl.get_present_current()
                self.publish_state(current_position, loadcell_value, current)
            return

        if self.r >= self.get_parameter('repeat').value * 2:
            self.dxl.disable_torque()
            self.get_logger().info(f'Task {self.selected_task} completed successfully')
            self.is_running = False
            self.is_sending_upper_limb_state = False
            return

        # Read sensors
        current_position = self.dxl.get_present_position()
        current_velocity = self.dxl.get_present_velocity() * 6  # 6 From KNU's firmware
        loadcell_value = self.read_loadcell()  # Implement according to your HX711 interface
        current = self.dxl.get_present_current()
        #  self.get_logger().info(f'direction: {self.direction}, velocity: {current_velocity}, positiion: {current_position}, theta: {self.theta}, current: {current}')
       
        # Calculate control
        self.current_time += 12.5
        
        if current_position >= 165 and self.direction == -1:
            self.direction = 1
            self.r += 1
            self.current_time = 0
        elif current_position <= 75 and self.direction == 1:
            self.direction = -1
            self.r += 1
            self.current_time = 0
            
        if self.direction > 0:
            self.theta = self.a * self.current_time / 1000 + 165
        else:
            self.theta = -self.a * self.current_time / 1000 + 75
            
        # Upperlimb_1DOF API task loop
        self.sensor_position = current_position - 90.0 # current_position - 90 degrees: convert robot coordinate to simulator coordinate
        self.sensor_velocity = current_velocity
        self.sensor_force = (loadcell_value + 50 - 300) * 9.8 / 1000  # Convert mg to N for API
        self.target_th = math.radians(self.theta - 90.0)  # theta - 90 degrees and convert to radians: convert robot coordinate to simulator corrdinate

        # Motion Task for temperary use
        if self.current_time % (2 * 1000) < 1000: # 1.0 sec flexion and 1.0 sec extension 
            self.target_th = math.radians(130)
        else:
            self.target_th = math.radians(5)

        self.previous_velocity = self.sensor_velocity
 
        if self.task_module: # If task module is loaded
            try:
                self.task_module.loop()
                # Get velocity command from upperlimb API
                velocityT = self.upperlimb.get_velocity_cmd()
                self.previous_velocity = self.sensor_velocity # new_1208
                #self.dxl.set_goal_velocity(velocityT / 6) # 6 From KNU's firmware
                self.dxl.set_goal_velocity(velocityT)
                self.get_logger().info(f'task current_time: {self.current_time}, velocity: {current_velocity}, position: {current_position}, target_th: {self.target_th}, theta: {self.theta}, velocity_cmd: {velocityT}')

                self.publish_state(current_position, loadcell_value, current)
                return 
            except Exception as e:
                self.get_logger().error(f'Error in task.loop(): {e}')
                # Fallback to original control logic

        # Impedance control
        delta_velocity = current_velocity - self.previous_velocity
        acceleration = delta_velocity / self.DELTA_TIME
        acceleration = max(min(acceleration, 500.0), -500.0) # TODO: Need to check the limit value
        self.previous_velocity = current_velocity

        delta_force = loadcell_value + 50 - 300
        velocityT = self.calculate_velocity(current_position, acceleration, delta_force, self.theta)
        self.dxl.set_goal_velocity(velocityT / 6) # 6 From KNU's firmware
        #  self.dxl.set_goal_velocity(velocityT)
        self.get_logger().info(f'robot velocity: {current_velocity}, position: {current_position}, theta: {self.theta}, velocity_cmd: {velocityT}')

        self.publish_state(current_position, loadcell_value, current)
    
    #joe Modify the speed on the flextion position at the Task 1    
    def calculate_velocity(self, position, acceleration, delta_force, theta):
        velocity = self.a * self.direction + (delta_force - self.m * acceleration - self.k * (position - theta)) / self.c
        if self.selected_task == 1 and self.direction == 1: # TASK 1 with Flextion
            velocity = -20 * self.direction + (delta_force - self.m * acceleration - self.k * (position - theta)) / self.c
        if self.selected_task == 2:  # TAhhSK 2 with Flextion
            velocity = -40 * self.direction + (delta_force - self.m * acceleration - self.k * (position - theta)) / self.c

        return max(min(velocity, 70.0), -70.0)
        
    def publish_state(self, position, loadcell_value, current):
        state = 'Flexion' if self.direction == 1 else 'Extension'

        state_msg = UpperLimbState()
        state_msg.state = state
        if self.r == -1:
            state_msg.repeat = 0
        else:
            state_msg.repeat = self.r
        state_msg.weight = loadcell_value
        state_msg.angle = position
        state_msg.current = float(current)

        #self.get_logger().info(f'State : {state}, Repeat : {self.r}, Weight : {loadcell_value:.2f}g, Elbow Angle : {position:.2f}°, Current : {current:.4f}mA')

        self.upper_limb_state_pub.publish(state_msg)
        
    def loadcell_callback(self, msg):
        self.loadcell_value = msg.data

    def read_loadcell(self):
        if self.loadcell_value is not None:
            return self.loadcell_value
        return 0.0  # TODO need error handling

    def handle_command(self, request, response):
        commands = {UpperLimbCommand.Request.COMMAND_START_STOP: 'COMMAND_START_STOP',
                    UpperLimbCommand.Request.COMMAND_RESET: 'COMMAND_RESET'}

        if request.command > UpperLimbCommand.Request.COMMAND_RESET:
            self.get_logger().error(f"Invalid command: {request.command}")
            response.success = False
            response.message = "Invalid command"
            return response
        self.get_logger().info(f"Received command: Task {request.task}, {commands[request.command]}")

        self.selected_task = request.task

        # Set control parameters by task
        if request.task == UpperLimbCommand.Request.TASK_1:
            self.m = 0.1
            self.c = 50
        elif request.task == UpperLimbCommand.Request.TASK_2:
            self.m = 0.1
            self.c = 15
        elif request.task == UpperLimbCommand.Request.TASK_3:
            self.c = 10000

        # Set control running status and reset variables
        if request.command == UpperLimbCommand.Request.COMMAND_START_STOP:
            self.is_running = not self.is_running
            if self.is_running:
                self.dxl.disable_torque() # Disable Dynamixel Torque
                self.dxl.set_operating_mode(OP_VELOCITY) # Set operating mode to velocity
                self.dxl.enable_torque()
        elif request.command == UpperLimbCommand.Request.COMMAND_RESET:
            if self.is_running:
                self.is_running = False
                time.sleep(2.0)
            self.is_sending_upper_limb_state = True
            self.current_time = 0.0
            self.direction = -1
            self.flag = 2
            self.r = -1
            self.start = 0
            self.theta = 0.0
            self.reset_motor_position()
        
        response.success = True
        response.message = "Command received"
        return response

def disable_torque():
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)
    
    if portHandler.openPort():
        print("Succeeded to open port")
    else:
        print("Failed to open port")
        return False
        
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change baudrate")
    else:
        print("Failed to change baudrate")
        return False

    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Disabled torque")
    portHandler.closePort()

    return dxl_error == 0
        
def main(args=None):
    rclpy.init(args=args)
    upper_limb_node = UpperLimbNode()

    try:
        rclpy.spin(upper_limb_node)
    except KeyboardInterrupt:
        upper_limb_node.get_logger().error('Keyboard Interrupt received')
    finally:
        upper_limb_node.get_logger().error('Finally')

        while not disable_torque(): # Retry until torque is disabled
            upper_limb_node.get_logger().error("Failed to disable torque, retrying...")
        rclpy.try_shutdown()

    disable_torque() # One more disable torque before exiting
    upper_limb_node.get_logger().info("Torque disabled!")

    upper_limb_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
