from .wearable_robot_api import Upperlimb_1DOF
import math

upperlimb = Upperlimb_1DOF()

def setup():
    upperlimb.init(rep_count=5, freq=60)

def loop():
    upperlimb.node.get_logger().info('task 2')
    current_position = -math.degrees(upperlimb.get_position()) 
    current_velocity = -upperlimb.get_velocity()
    current_force = upperlimb.get_force()
    delta_velocity = current_velocity - upperlimb.get_previous_velocity()
    acceleration = delta_velocity / upperlimb.DELTA_TIME
    acceleration = max(min(acceleration, 500.0), -500.0)
    delta_force = current_force * 1000 / 9.8   # dimension change
    m = upperlimb.get_moment() # moment of inertia 관성 계수
    k = upperlimb.get_spring() # spring 계수
    c = upperlimb.get_damping() # damping 계수

    if upperlimb.get_target_angle() > 90:
        velocity = -0.6 + (delta_force - m * acceleration - k * (current_position - upperlimb.get_target_angle())) / c # flexion
        upperlimb.node.get_logger().info('flexion')
    else:
        velocity = 1.2  # extension
        upperlimb.node.get_logger().info('extension')
    upperlimb.set_velocity(velocity)
    upperlimb.node.get_logger().info(f'[Task] Target Angle: {upperlimb.get_target_angle():.2f}, current_position: {current_position:.2f}, current_velocity: {current_velocity:.2f}, current_force: {current_force:.2f}')
