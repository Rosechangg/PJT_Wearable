# upperlimb instance will be set by elbow_vel_cmd.py
upperlimb = None

def setup():
    upperlimb.init(rep_count=20, freq=60)

def loop():
    # task1
    if upperlimb.get_target_angle() > 90:
        upperlimb.set_velocity(-1.2)
    else:
        upperlimb.set_velocity(1.2)
