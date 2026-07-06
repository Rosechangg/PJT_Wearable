import math


class Upperlimb_1DOF:
    """API class for controlling the upper limb exoskeleton (Singleton)"""

    _instance = None

    def __new__(cls, node=None):
        if cls._instance is None:
            cls._instance = super(Upperlimb_1DOF, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, node=None):
        if self._initialized:
            if node is not None:
                self.node = node
            return

        self.node = node
        self.rep_count = 0
        self.freq = 60
        self.velocity_cmd = 0.0
        self.DELTA_TIME = self.node.DELTA_TIME if self.node else 0.02 # Use node's DELTA_TIME if available, else default to 0.02s
        self._initialized = True

    def init(self, rep_count=20, freq=60):
        """Initialize the upperlimb control parameters"""
        self.rep_count = rep_count
        self.freq = freq
        if self.node:
            self.node.get_logger().info(f'Upperlimb_1DOF initialized: rep_count={rep_count}, freq={freq}')
            self.node.declare_parameter('repeat', rep_count + 1)

    def get_target_angle(self):
        """Get the current target angle in degrees"""
        if self.node:
            return math.degrees(self.node.target_th)
        return 0.0

    def set_velocity(self, velocity):
        """Set the velocity command"""
        self.velocity_cmd = velocity

    def get_velocity_cmd(self):
        """Get the current velocity command"""
        return self.velocity_cmd

    def get_velocity(self):
        """Get the current sensor velocity"""
        if self.node:
            return self.node.sensor_velocity
        return 0.0

    def get_previous_velocity(self):
        """Get the previous sensor velocity"""
        if self.node:
            return self.node.previous_velocity
        return 0.0

    def set_previous_velocity(self, velocity):
        """Set the previous sensor velocity"""
        if self.node:
            self.node.previous_velocity = velocity

    def get_force(self):
        """Get the current sensor force"""
        if self.node:
            return self.node.sensor_force
        return 0.0

    def get_position(self):
        """Get the current sensor position"""
        if self.node:
            return self.node.sensor_position
        return 0.0

    def get_moment(self):
        """Get the moment of inertia (관성 계수)"""
        return 0.0  # Default value, can be configured

    def get_spring(self):
        """Get the spring coefficient (spring 계수)"""
        return 0.0  # Default value, can be configured

    def get_damping(self):
        """Get the damping coefficient (damping 계수)"""
        return 100.0  # Default value, same as in elbow_vel_cmd control_logic

    @property
    def target_th(self):
        """Get the target angle in radians"""
        if self.node:
            return self.node.target_th
        return 0.0
