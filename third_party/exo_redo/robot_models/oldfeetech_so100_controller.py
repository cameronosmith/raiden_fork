import numpy as np
from feetech import FeetechMotorsBusConfig,FeetechMotorsBus,find_port
import pickle
import matplotlib.pyplot as plt
import os,time,sys
import mujoco
sys.path.append(".")
from ExoConfigs.so100_holemounts import SO100_CONFIG
from exo_utils import get_link_poses_from_robot, position_exoskeleton_meshes

# pybullet helpers
targ_joint_state_to_match=np.array([0, -1.57, 1.57, 1.57, -1.57, 0]) 
rest_pos=np.array([ 0.05215535,-3.24357304, 3.15767012, 0.5161552, -4.4, -0.21782527])
sensor_to_rad=2*np.pi/4096
sensor_to_offset = lambda sim_state,signs,sensor_state: sim_state/(signs*sensor_to_rad) - sensor_state

class Arm: 

    def __init__(self,calib=None):
        # pytorch kinematics calib for arm
        self.last_solve_state=[0]*6
        self.calib=calib
        self.motors_connected=False


        self.model = mujoco.MjModel.from_xml_string(SO100_CONFIG.xml) 
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.data.ctrl[:] = targ_joint_state_to_match # home position

        if calib is not None: # store calib or do calibration
            self.calib["port"]= "/dev/tty.usbserial-0001"
            self.connect_motors()
            try: 
                self.connect_motors()
                self.motors_connected=True
            except:
                print("Error connecting motors")
                pass
        else: 
            self.calib = {} # dict for calib we will store as config
            self.calibrate()
        self.calib["sensor_offset_raw"]=np.copy(self.calib["sensor_offset"])

    # turn motors on
    def connect_motors(self): 
        self.motors = [FeetechMotorsBus( FeetechMotorsBusConfig( port=self.calib["port"], motors={motor_name: (motor_i+1, "sts3215")}) ) 
                                    for motor_i,motor_name in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]) ]
        for motor in self.motors: motor.connect();motor.write("Present_Speed", 4);motor.write("Acceleration",  4)
    # get position of all joints
    def get_pos(self,raw=False,suf=""): 
        return np.array([self.get_pos_i(i,suf=suf) if not raw else self.motors[i].read("Present_Position") for i in range(len(self.motors))]).squeeze(1)
    # get position of a specific joints
    def get_pos_i(self,i,suf=""): return (self.motors[i].read("Present_Position") + self.calib["sensor_offset%s"%suf][i] ) * (self.calib["signs"][i]*sensor_to_rad)
    # get position of all joints
    def write_pos(self,calibrated_positions): 
        for i,calib_pos in enumerate(calibrated_positions): self.write_pos_i(calib_pos,i)
    # get position of a specific joint
    def write_pos_i(self,calib_pos,i): 
        uncalib_pos = calib_pos/(self.calib["signs"][i]*sensor_to_rad)- self.calib["sensor_offset"][i]
        self.motors[i].write("Goal_Position", int(uncalib_pos))
    # turn off motors
    def disconnect(self): [motor.write(x,0) for x in ["Torque_Enable","Present_Speed","Acceleration"] for motor in self.motors] # turn off motors

    # Calibration scripts
    def calibrate(self):

        # 1. Get port of arm
        if 0:
            self.calib["port"] = str(find_port())
            input("Found port %s. Press any key after plugging port back in"%self.calib["port"])
            for _ in range(15):print("NOTE USING HARCDODED PORT")
        else:
            #self.calib["port"]= "/dev/tty.usbmodem59700733101"
            #self.calib["port"]= "/dev/tty.usbmodem5A680121681"
            self.calib["port"]= "/dev/tty.usbmodem5A7C1216291"

        # 2. Connect motors
        self.connect_motors();print("Motors successfully connected") 
        # Placeholder vals
        self.calib["signs"]=np.ones(len(targ_joint_state_to_match)) 
        self.calib["sensor_offset"]= sensor_to_offset(targ_joint_state_to_match,self.calib["signs"],self.get_pos(raw=True)) 

        # 3. Get signs (whether positive on sensor state is positive on sim robot state) by rendering in sim and reversing backward states
        self.write_state=False
        def key_callback(keycode):
            key_char = chr(keycode).lower() if keycode < 256 else None
            if key_char == ' ': 
                self.write_state=True
                self.calib["sensor_offset"]= sensor_to_offset(targ_joint_state_to_match,self.calib["signs"],self.get_pos(raw=True)) 
                time.sleep(0.1)
                print("Move robot joints and press CTRL+Number (1-6) to toggle sign for that joint")
                print(f"Current signs: {self.calib['signs']}")
            elif key_char in [str(i) for i in range(1, 7)]:
                joint_idx = int(key_char) - 1
                self.calib["signs"][joint_idx] *= -1
                print(f"Toggled joint {joint_idx+1} (Option+{key_char}). Signs: {self.calib['signs']}")
            elif key_char == 'q': 
                self.write_state=None
        viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False, key_callback=key_callback)
        while viewer.is_running():
            if self.write_state:self.data.qpos[:] = self.data.ctrl[:] = self.get_pos()
            position_exoskeleton_meshes(SO100_CONFIG, self.model, self.data, get_link_poses_from_robot(SO100_CONFIG, self.model, self.data))
            print(self.get_pos(),self.write_state)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            if self.write_state is None:break
        self.calib["sensor_offset"]= sensor_to_offset(targ_joint_state_to_match,self.calib["signs"],self.get_pos(raw=True)) 
       
if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="A simple example")
    parser.add_argument("--arm_config",'-c',default=None) # no gui, just for debugging/printing
    parser.add_argument("--cam_vis",action="store_true") # cam rerender but dont reset config
    parser.add_argument("--live_render",action="store_true") # just render arm in sim
    parser.add_argument("--save_imgs",action="store_true") # save livestreamed imgs
    args = parser.parse_args()

    # If already connected just view arm in sim driven by real arm 
    if os.path.exists(str(args.arm_config)): 
        self=arm=Arm( pickle.load(open(args.arm_config, 'rb')) )
        viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        while viewer.is_running():
            import pdb;pdb.set_trace()
            self.data.qpos[:] = self.data.ctrl[:] = self.get_pos()
            print(self.get_pos())
            position_exoskeleton_meshes(SO100_CONFIG, self.model, self.data, get_link_poses_from_robot(SO100_CONFIG, self.model, self.data))
            mujoco.mj_step(self.model, self.data)
            viewer.sync()

        arm.disconnect()
    else: # do arm calibration
        arm=Arm()
        pickle.dump(arm.calib, open(args.arm_config, 'wb'))
