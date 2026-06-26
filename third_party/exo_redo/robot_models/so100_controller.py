import numpy as np
import pickle
import matplotlib.pyplot as plt
import os,time,sys
import mujoco
sys.path.append(".")
from ExoConfigs.so100_adhesive import SO100AdhesiveConfig
from exo_utils import get_link_poses_from_robot, position_exoskeleton_meshes
sys.path.append("/Users/cameronsmith/Projects/robotics_testing/servo_testing/FTServo_Python")

from scservo_sdk import PortHandler, sms_sts
from exo_utils import estimate_robot_state, detect_and_set_link_poses, position_exoskeleton_meshes, render_from_camera_pose, get_link_poses_from_robot

# pybullet helpers
targ_joint_state_to_match=np.array([0, -1.57, 1.57, 1.57, 1.57, 0]) 
rest_pos = np.array( [0.21015537,2.83405884,3.14539827,4.23605861,1.58073787,1.63400049])
sensor_to_rad=2*np.pi/4096
sensor_to_offset = lambda sim_state,signs,sensor_state: sim_state/(signs*sensor_to_rad) - sensor_state

class Arm: 

    def __init__(self,calib=None):
        self.last_solve_state=[0]*6
        self.calib=calib
        self.motors_connected=False

        self.model = mujoco.MjModel.from_xml_string(SO100AdhesiveConfig().xml) 
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.data.ctrl[:] = targ_joint_state_to_match # home position

        self.port = PortHandler("/dev/tty.usbserial-0001")
        if not self.port.openPort(): sys.exit(f"Failed to open port")
        if not self.port.setBaudRate(115200): sys.exit(f"Failed to set baud")
        self.sts = sms_sts(self.port)

        ping=self.sts.ping(1)
        print(ping)
        if ping[1]==0 and ping[2]==0: print("Motors successfully connected");self.motors_connected=True
        else: sys.exit(f"Failed to connect to motors");
        print("consider adding temperature check here to aoiv overheating")
        
        # Set torque limits to reduce mechanical vibration

        if calib is None: 
            self.calib = {} # dict for calib we will store as config
            self.calibrate()
        self.calib["sensor_offset_raw"]=np.copy(self.calib["sensor_offset"])

    # get position of all joints
    def get_pos(self,raw=False): 
        raw_pos = np.array([self.sts.ReadPos(i+1)[0] for i in range(6)])
        
        if raw:
            return raw_pos
        else:
            calibrated_pos = (raw_pos + self.calib["sensor_offset"] ) * self.calib["signs"]*sensor_to_rad
            # Allow negative values - no wrap-around
            return calibrated_pos
    def write_pos(self,calibrated_positions,slow=False,skip_ids=[]): 
        for i,pos in enumerate( calibrated_positions/(self.calib["signs"]*sensor_to_rad)-self.calib["sensor_offset"] ): 
            if i in skip_ids: continue
            pos=pos%4096
            if pos<0: pos+=4096
            
            # Balanced speed/accel for cheaper feetech motors - fast enough but stable
            speed = 250 if not slow else 80
            accel = 100 if not slow else 50
            self.sts.RegWritePosEx(i+1,int(pos),speed,accel)
        self.sts.RegAction()
    
    def emergency_stop(self):
        """Emergency stop - disable torque on all motors to kill motion"""
        for i in range(6): self.sts.write1ByteTxRx(i+1, 40, 0)
    
    def set_middle_position(self, servo_ids=None):
        ADDR_STS_STEP_MIDDLE = 40
        STEP_MIDDLE_CMD = 128
        if servo_ids is None: servo_ids = list(range(1, 7))  # Servos 1-6
        for servo_id in servo_ids:
            comm_result, error = self.sts.write1ByteTxRx(servo_id, ADDR_STS_STEP_MIDDLE, STEP_MIDDLE_CMD)
            if comm_result != 0:  # 0 indicates COMM_SUCCESS
                print(f"Warning: Comm error setting middle position for servo {servo_id}: {comm_result}")
            elif error != 0:
                print(f"Warning: Servo error setting middle position for servo {servo_id}: {error}")
            else:
                print(f"Middle position set for servo {servo_id}")
    
    def calibrate(self): # Assumes robot is in target state above (see target joint state above / middle position). Just manually change signs here if joint is backwards in viewer
        self.calib["signs"]=np.array([-1,1,1,1,1,1]).astype(float)
        self.calib["sensor_offset"]= sensor_to_offset(targ_joint_state_to_match,self.calib["signs"],self.get_pos(raw=True)) 
        print(self.calib["sensor_offset"])
       
if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser(description="A simple example")
    parser.add_argument("--arm_config",'-c',default=None) # no gui, just for debugging/printing
    parser.add_argument("--cam_vis",action="store_true") # cam rerender but dont reset config
    parser.add_argument("--control_test",action="store_true") # move arm to some position
    parser.add_argument("--recalib_from_img",action="store_true") 
    parser.add_argument("--save_imgs",action="store_true") # save livestreamed imgs
    args = parser.parse_args()

    # If already connected just view arm in sim driven by real arm 
    if not os.path.exists(str(args.arm_config)): 
        arm=Arm()
        pickle.dump(arm.calib, open(args.arm_config, 'wb'))
    self=arm=Arm( pickle.load(open(args.arm_config, 'rb')) )


    while True:print(self.get_pos(raw=True),self.get_pos())

    if args.control_test:
        robot_pos = self.get_pos()
        targ_positions=[]
        targ_robot_pos = targ_joint_state_to_match.copy()

        targ_robot_pos=np.array([-0.00268742, -1.6865245,   1.65632287,  1.51128661,  1.55649603,  1.34])

        targ_positions.append(targ_robot_pos)
        targ_positions.append(rest_pos)

        last_pos = self.get_pos()
        done_moving=False

        while True:
            curr_pos = self.get_pos()
            #done_moving=np.max(np.abs(curr_pos-last_pos))<0.01
            sts_action_lists=[self.sts.ReadMoving(i+1)[0] for i in range(6)]
            done_moving=not any(sts_action_lists)
            last_pos = curr_pos
            if done_moving:
                if len(targ_positions)==0: break
                curr_action=targ_positions.pop(0)
                print("writing new action")
                self.write_pos(curr_action,slow=False)
        print("done moving")
        self.emergency_stop()
    elif args.recalib_from_img:
        cam_K=None#np.array([[1.19087964e+03, 0.00000000e+00, 9.59500000e+02], [0.00000000e+00, 1.19087964e+03, 5.39500000e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        import cv2
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            print("reading frame")
            if not ret: print ("Failed to read frame from camera");continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try: link_poses, camera_pose_world, cam_K, corners_cache,corners_vis,obj_img_pts = detect_and_set_link_poses(rgb, self.model, self.data, SO100AdhesiveConfig,cam_K=cam_K)
            except: print("pose est error");continue
            configuration = estimate_robot_state( self.model, self.data, SO100AdhesiveConfig(), link_poses, ik_iterations=15)
            self.data.qpos[:] = self.data.ctrl[:] = configuration.q
            mujoco.mj_forward(self.model, self.data)
            rendered = render_from_camera_pose(self.model, self.data, camera_pose_world, cam_K, *rgb.shape[:2])
            overlay = (rgb.astype(float) * 0.5 + rendered.astype(float)  * 0.5).astype(np.uint8)
            display = np.hstack([corners_vis, rendered, overlay])

            cv2.imshow("display", display[...,::-1])
            waitkey=cv2.waitKey(1)& 0xFF
            if waitkey==ord('s'): 
                img_pos=configuration.q
                #img_pos[-1]=1.5
                self.calib["sensor_offset"] = sensor_to_offset(img_pos,self.calib["signs"],self.get_pos(raw=True))
                new_arm_config_path = args.arm_config.replace(".pkl", "_fromimg.pkl")
                print(f"Saved calibration to {new_arm_config_path}")
                pickle.dump(self.calib, open(new_arm_config_path, 'wb'))
            elif waitkey==ord('q'): break
    else: # do arm calibration
        viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        while viewer.is_running():
            robot_pos,raw_pos = self.get_pos(),self.get_pos(raw=True)
            self.data.qpos[:] = self.data.ctrl[:] = robot_pos
            print(robot_pos,raw_pos)
            position_exoskeleton_meshes(SO100AdhesiveConfig(), self.model, self.data, get_link_poses_from_robot(SO100AdhesiveConfig(), self.model, self.data))
            mujoco.mj_step(self.model, self.data)
            viewer.sync()

