import Upperlimb_1DOF

upperlimb = Upperlimb_1DOF()

def setup():
    upperlimb.init(rep_count=20, freq=60)

def loop():
    delta_velocity = upperlimb.get_velocity() - upperlimb.get_previous_velocity()
    acceleration = delta_velocity / upperlimb.DELTA_TIME
    acceleration = max(min(acceleration, 500.0), -500.0)
    upperlimb.previous_velocity = upperlimb.get_velocity()
    delta_force = upperlimb.get_force() * 1000 / 9.8   # dimension change
    m = upperlimb.get_moment() # moment of inertia 관성 계수
    k = upperlimb.get_spring() # spring 계수
    c = upperlimb.get_damping() # damping 계수

    if upperlimb.get_target_angle() > 90:
        velocity = -0.6 + (delta_force - m * acceleration - k * (upperlimb.get_position - upperlimb.target_th)) / c # flexion
    else:
        velocity = 1.2  # extension

    previous_velocity = upperlimb.get_velocity()