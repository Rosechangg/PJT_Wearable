import numpy as np
import mujoco
import cvxpy as cp
import json
import xml.etree.ElementTree as ET

import os
from ament_index_python.packages import get_package_share_directory
share_dir = get_package_share_directory("wearable_robot_mujoco")
patient_models = os.path.join(share_dir, "config", "patient_models.json")
#  patient_models = "patient_models.json"

class ElbowMuscleBrain():
    
    def __init__(self, PModel, kp=5.0, kd=1.0, xml_path="myoelbow_1dof6muscles.xml"):
        super(ElbowMuscleBrain, self).__init__()

        self.vel_cmd = 0.0

        # Load MuJoCo model 
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        self.ctrl_period = 0.02

        # Patient parameters
        self._load_model_from_json(patient_models, PModel)

        # trajectory planning parameters
        self.current_time = 0.0
        self.switch_interval = 2.0
        self.target_th = np.deg2rad(130)
        self.q_goal = self.target_th
        self.trajectory_angle = 0.0
        self.trajectory_velocity = 0.0

        self.zeta = 1.0      # Critical damping (no overshoot)
        self.T_rise = 1.5    # Want to rise in ~2s
        self.omega_n = 4.0 / self.T_rise  # heuristic: 4/T_rise for 98% in T_rise

        # control parameters
        self.prev_err = 0.0
        self.d_hat = np.zeros((6,1))
        self.act = np.zeros((6,1))
        self.beta = 1.0

        self.kp = kp
        self.kd = kd

        # quantization
        self.n = 10

        # Muscle parameter
        self.rest_length = np.copy(self.data.actuator_length)

        self._init_patient_model()

        # device joint parameter
        self.jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "exo_hinge")
        self.sid_force = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "force_sensor")

        # device motor parameter
        self.aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Exo")
        self.gear = float(self.model.actuator_gear[self.aid,0])

        # device sensor parameter
        self.sensor_dist = self._get_sensor_dist()

    def set_velocity_command(self, vel_cmd):
        self.vel_cmd = vel_cmd

    def _init_patient_model(self):
        # Read XML and read user attributes from <body> tag
        xml_dir = os.path.dirname(self.xml_path)
        user_xml_path = os.path.join(xml_dir, "assets", "myoelbow_1dof6muscles_1dofexo_body_revised_2.xml")
        print(user_xml_path)
        root = ET.parse(user_xml_path).getroot()
        body_ = root.find("worldbody").find("body").find("body").find("body")
        user_attrib = body_.attrib.get('user').split(' ')
        print(f"User attribute: {user_attrib}")

        self.V_gain = float(user_attrib[0])
        self.beta = float(user_attrib[1])
        self.RI_ratio = float(user_attrib[2])
        self.n = int(user_attrib[3])
        R_gain = float(user_attrib[4])
        self.Fmax = float(user_attrib[5])
        self.K_pass = float(user_attrib[6])
        self.L_opt = float(user_attrib[7])

        # if R_gain < 0.3:
        #     self.R_gain = np.array([R_gain] * 6)
        # else:
        #     self.R_gain = np.array([
        #         0.3, 0.3, 0.3,       # Triceps (TRIlong, TRIlat, TRImed)
        #         R_gain, R_gain, R_gain  # Biceps (BIClong, BICshort, BRA)
        #     ])
        self.R_gain = np.array([R_gain] * 6)

        triceps_ids = [0, 1, 2]
        biceps_ids = [3, 4, 5]

        # at first, save actuator parameter
        if not hasattr(self, "_base_gainprm"):
            self._base_gainprm = np.copy(self.model.actuator_gainprm)
            self._base_dynprm = np.copy(self.model.actuator_dynprm)
            self._base_ctrlrange = np.copy(self.model.actuator_ctrlrange)
            self._base_lengthrange = np.copy(self.model.actuator_lengthrange)

        for i in [0,1,2,3,4,5]:
            # reset to base first
            self.model.actuator_gainprm[i][2] = self._base_gainprm[i][2] * self.Fmax
            self.model.actuator_dynprm[i][0] = self._base_dynprm[i][0] * (2 * self.K_pass)
            self.model.actuator_ctrlrange[i][1] = self._base_ctrlrange[i][1] * self.L_opt
            self.model.actuator_lengthrange[i][1] = self._base_lengthrange[i][1] * self.L_opt
        
        # L_opt_scale = self.L_opt
        # F_max_scale = self.Fmax
        # K_pass_scale = 2 * self.K_pass

        # for i in biceps_ids:
        #     self.model.actuator_gainprm[i][0] *= F_max_scale    # F_max
        #     self.model.actuator_dynprm[i][0] *= K_pass_scale        # K_pass
        #     self.model.actuator_ctrlrange[i][1] *= L_opt_scale  # range max
        #     self.model.actuator_lengthrange[i][1] *= L_opt_scale  # lmax 

    def _load_model_from_json(self, json_path, model_name):

        with open(json_path, 'r') as f:
            model_data = json.load(f)

        if model_name not in model_data:
            raise ValueError(f"Model '{model_name}' not found in the JSON file.")

        params = model_data[model_name]

        self.V_gain = params["V_gain"]
        self.RI_ratio = params["RI_ratio"]
        self.K_pass = params["K_pass"]
        self.L_opt = params["L_opt"]
        self.Fmax = params["Fmax"]

        R_gain = params["R_gain"]
        if R_gain < 0.3:
            self.R_gain = np.array([R_gain] * 6)
        else:
            self.R_gain = np.array([
                0.3, 0.3, 0.3,       # Triceps (TRIlong, TRIlat, TRImed)
                R_gain, R_gain, R_gain  # Biceps (BIClong, BICshort, BRA)
            ])

    def _update_desired_angle(self):

        alpha = 0.5
        
        # rythm
        if self.current_time % (2 * self.switch_interval) < self.switch_interval:
            self.target_th = np.deg2rad(130)
        else:
            self.target_th = np.deg2rad(5)
        
        # trajectory generation
        acc = self.omega_n ** 2 * (self.target_th - self.trajectory_angle) - 2 * self.zeta * self.omega_n * self.trajectory_velocity
        self.trajectory_velocity += acc * self.ctrl_period
        self.trajectory_angle += self.trajectory_velocity * self.ctrl_period

        q = self.data.qpos[0]
        qdot = self.data.qvel[0]

        # hybrid generation
        q_des = (1 - alpha) * q + alpha * self.trajectory_angle
        qdot_des = (1 - alpha) * qdot + alpha * self.trajectory_velocity
        qddot_des = acc

        self.current_time += self.ctrl_period
        self.q_des_cache = (q_des, qdot_des, qddot_des)
        return q_des

    def PMC_SMA(self):

        return self._update_desired_angle()

    def _Striatum(self, target_angle):
        # PD controller        
        err_p = target_angle - self.data.qpos[0]
        err_d = (err_p - self.prev_err) / self.ctrl_period
        target_torque = self.kp*err_p + self.kd*err_d + self.beta * (self.data.qfrc_bias[0] - self.data.qfrc_passive[0])
        
        self.prev_err = err_p

        return target_torque

    def _GPi(self, target_torque, z=None):
        # muscel configuration
        Ta, Tp = self._compute_muscle_torques(self.model, self.data)
        Ta = Ta.reshape(1, -1)  # (1x6 matrix)

        lambda_reg = 1e-2   # Regularization
        gamma = 0.005       # Co-contraction gain (negative to encourage)
        gamma_ri = 2.0    # Soft RI penalty
        
        I3 = np.eye(3)
        O3 = np.ones((3, 3))
        k = (1.0 - self.RI_ratio) / 3.0
        A = np.block([
            [I3,       k * O3],
            [k * O3,   I3    ]
        ])


        # Optimization variables
        a = cp.Variable((6, 1))
        a_eff = A @ a

        # QP optimization: Least Squares + co-contraction
        objective = cp.Minimize(
            cp.norm2(Ta @ a_eff - (target_torque - Tp))**2 + lambda_reg * cp.norm2(a)**2 # + gamma_ri * (1.0 - self.RI_ratio) * cp.square(cp.sum(a[0:3]) -  cp.sum(a[3:6]))  # Co-contraction term
        )

        # Constraints (0 <= a <= 1)
        constraints = [
            0 <= a, 
            a <= 1
        ]

        # QP solving
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, eps_abs=1e-4, eps_rel=1e-4)

        # Solution
        a_opt = np.clip(A @a.value, 0.0, 1.0)

        if a_opt is None:
            a_opt = np.zeros((6, 1))
        
        a_opt = self._quantize_pow2(a_opt, self.n)

        return a_opt

    def Basal_Ganglia(self, target_angle):
        target_torque = self._Striatum(target_angle)
        act = self._GPi(target_torque)

        return act, target_torque

    def Thalamus(self, act):
        # To Do: design filter
        # tau = 0.01
        # alpha = (self.ctrl_period / (self.ctrl_period + tau))

        # act_filtered = alpha * act + (1 - alpha) * self.act
        act_filtered = act
        # self.act = act_filtered

        return act_filtered

    def M1(self, act_filtered):

        # u_ff = self.V_gain * act_filtered
        u_ff = np.clip(act_filtered, 0.0, self.V_gain)

        return u_ff

    def Cerebrum(self, u_ff):
        # To Do: DOB design
        alpha = 1.0 # 1.0 default
        a_meas = self.data.act[0:6].copy().reshape((6, 1))

        # estimated disturbance
        raw_error = u_ff - a_meas

        # Low-pass filtering (1st order)
        beta = self.ctrl_period / (0.001 + self.ctrl_period)
        self.d_hat += beta * (raw_error - self.d_hat)

        # compensation
        u_fb = alpha * self.d_hat

        return u_fb

    def _Reflex(self):

        k_v = 1.0   # velocity sensitivity
        k_l = 0.1   # length sensitivity

        Ia_aff = np.zeros((6,1))

        for i in range(6):
            L = self.data.actuator_length[i]
            dL = self.data.actuator_velocity[i]
            L0 = self.rest_length[i]

            Ia_aff[i] = self.R_gain[i] * (k_v * dL + k_l * (L - L0))

        return Ia_aff

    def Spinal_cord(self, u_ff, u_fb):
        # neural integration
        # u_integrated = u_ff + u_fb + self._Reflex()
        u_integrated = u_ff + self._Reflex()

        return u_integrated
    
    ################## Functions related to Device Motions  ##################
    
    def sensor_position(self):
        adr = self.model.jnt_qposadr[self.jid]
        reading = self.data.qpos[adr]

        return reading
    
    def sensor_velocity(self):
        adr = self.model.jnt_dofadr[self.jid]
        reading = self.data.qvel[adr]

        return reading
    
    def sensor_force(self):  

        return self.sensor_dist * self._sensor_torque()

    def _sensor_torque(self):
        adr = self.model.jnt_dofadr[self.jid]
        reading = self.data.qfrc_passive[adr]   # Tendon은 passive component로 분류됨

        return reading

    def simulation_time(self):
        return self.current_time
   
    def control_logic(self):
        if self.target_th > np.deg2rad(90): # Task에 따라서(flexion, extension) 토크를 주는 간단한 동작
            velocity = -1.2
        else:
            velocity = 1.2

        return velocity

    def _compensator(self):
        adr = self.model.jnt_dofadr[self.jid]
        p = self.data.qfrc_passive[adr] 
        b = self.data.qfrc_bias[adr] 

        return p-b

    def _get_sensor_dist(self):
        mujoco.mj_forward(self.model, self.data)  # 위치/회전 업데이트

        # 사이트 ID 얻기
        sid_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_joint")
        sid_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_sensor_point")

        # 월드 좌표의 사이트 위치
        p_joint = self.data.site_xpos[sid_joint].copy()
        p_sensor = self.data.site_xpos[sid_sensor].copy()

        # 벡터/거리
        r_vec = p_sensor - p_joint
        dist = float(np.linalg.norm(r_vec))

        return dist


    ###########################     Simulation     ###########################

    def step(self):
        # elbow control mechanism (Human)
        target_angle = self.PMC_SMA()
        act, target_torque = self.Basal_Ganglia(target_angle)
        act_filt = self.Thalamus(act)
        u_ff = self.M1(act_filt)
        u_fb = self.Cerebrum(u_ff)
        u_integrated = self.Spinal_cord(u_ff, u_fb)

        # elbow control mechanism (Exo velocity control)
        vel_cmd = self.vel_cmd 
        
        # execution
        self.data.ctrl[0:6] = np.clip(u_integrated.flatten(), 0.0, 1.0)         
        self.data.ctrl[7] = vel_cmd

        for _ in range(int(self.ctrl_period/self.model.opt.timestep)):  # controller sampling time에 맞게 
            self.data.ctrl[6] = -self._compensator()                    # 모터 내부 연산이므로 빠르게 업데이트                      
            mujoco.mj_step(self.model, self.data)        

        current_angle = self.data.qpos[0]

        # verbose
        info = {
            "target_angle": self.target_th,
            "current_angle": current_angle,
            "input": target_torque,
            "actuation": self.data.qfrc_actuator[0],
            "TRIlong": self.data.ctrl[0],
            "TRIlat":  self.data.ctrl[1],
            "TRImed":  self.data.ctrl[2],
            "BIClong": self.data.ctrl[3],
            "BICshort":self.data.ctrl[4],
            "BRA":     self.data.ctrl[5]
        }

        return info

    def close(self):
        pass

    def _compute_muscle_torques(self, model, data):
        num_muscles = 6 # nu: # of actuator
        Ta = np.zeros(num_muscles) 

        # current status of activation
        a_m = np.copy(data.act)

        # passive torque
        data.act.fill(0) 
        mujoco.mj_forward(model, data) 
        Tp = data.qfrc_actuator[0] 

        # active torque
        for muscle_id in range(num_muscles):
            # muscle by muscle activation
            data.act.fill(0)
            data.act[muscle_id] = 1.0
            mujoco.mj_forward(model, data)

            # active torque = total torque - passive torque
            Ta[muscle_id] = data.qfrc_actuator[0] - Tp

        # activation restoration
        data.act[:] = a_m

        return Ta, Tp

    def set_patient_model(self, params):
        self.V_gain = params["V_gain"]
        self.R_gain = np.array([0.3, 0.3, 0.3, params["R_gain"], params["R_gain"], params["R_gain"]]) \
            if params["R_gain"] >= 0.3 else np.array([params["R_gain"]] * 6)
        self.RI_ratio = params["RI_ratio"]
        self.K_pass = params["K_pass"]
        self.L_opt = params["L_opt"]
        self.Fmax = params["Fmax"]
        self._init_patient_model()

    def _quantize_pow2(self, a, n, return_index=False):
        """
        a: [0,1] 범위의 ndarray (예: QP 해; shape (6,) 등)
        n: 단계 수를 2^n으로 설정 (예: n=3 -> 8단계)
        return_index: True면 (a_q, idx) 모두 반환. idx는 0..(2^n-1)

        방식: 0과 1을 포함하는 mid-tread 균등 양자화.
        """
        a = np.asarray(a, dtype=float)
        L = 1 << n                 # 2^n
        if L < 2:
            raise ValueError("n은 1 이상이어야 합니다.")

        delta = 1.0 / (L - 1)      # 0..1 포함하려면 (L-1)로 나눔
        idx = np.rint(a / delta)   # 가장 가까운 레벨로 반올림
        idx = np.clip(idx, 0, L-1) # 인덱스 범위 제한
        a_q = idx * delta          # 양자화된 값

        return (a_q, idx.astype(int)) if return_index else a_q
