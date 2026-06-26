#!/usr/bin/env python3
"""
Feetech Servo Calibration Tool

This script helps calibrate Feetech servos by:
1. Asking user to move each joint to its minimum safe position
2. Asking user to move each joint to its maximum safe position  
3. Setting servo limits, offset, and phase based on these positions
4. Saving calibration data for future use
"""

import sys
import os
import pickle
import numpy as np
sys.path.append(".")
from ExoConfigs.so100_holemounts import SO100_CONFIG
sys.path.append("/Users/cameronsmith/Projects/robotics_testing/servo_testing/FTServo_Python")

from scservo_sdk import PortHandler, sms_sts

class FeetechCalibration:
    def __init__(self):
        self.port = PortHandler("/dev/tty.usbserial-0001")
        if not self.port.openPort(): 
            sys.exit(f"Failed to open port")
        if not self.port.setBaudRate(115200): 
            sys.exit(f"Failed to set baud")
        self.sts = sms_sts(self.port)
        
        # Test connection
        ping = self.sts.ping(1)
        if ping[1] == 0 and ping[2] == 0:
            print("Motors successfully connected")
        else:
            sys.exit(f"Failed to connect to motors")
        
        self.num_joints = 6
        self.calibration_data = {}
        
    def read_position(self, joint_id):
        """Read current position of a joint"""
        return self.sts.ReadPos(joint_id)[0]
    
    def set_servo_limits(self, joint_id, min_pos, max_pos):
        """Set servo angle limits"""
        print(f"Setting limits for joint {joint_id}: min={min_pos}, max={max_pos}")
        self.sts.write2ByteTxRx(joint_id, 9, min_pos)   # Min_Angle_Limit
        self.sts.write2ByteTxRx(joint_id, 11, max_pos)  # Max_Angle_Limit
    
    def set_servo_offset(self, joint_id, offset):
        """Set servo zero offset"""
        print(f"Setting offset for joint {joint_id}: {offset}")
        self.sts.write2ByteTxRx(joint_id, 31, offset)
    
    def set_servo_phase(self, joint_id, phase):
        """Set servo phase (direction)"""
        print(f"Setting phase for joint {joint_id}: {phase}")
        self.sts.write1ByteTxRx(joint_id, 18, phase)
    
    def calibrate_joint(self, joint_id):
        """Calibrate a single joint"""
        print(f"\n=== Calibrating Joint {joint_id} ===")
        print(f"Current position: {self.read_position(joint_id)}")
        
        # Get minimum position
        input(f"Move joint {joint_id} to its MINIMUM safe position, then press Enter...")
        min_pos = self.read_position(joint_id)
        print(f"Minimum position recorded: {min_pos}")
        
        # Get maximum position  
        input(f"Move joint {joint_id} to its MAXIMUM safe position, then press Enter...")
        max_pos = self.read_position(joint_id)
        print(f"Maximum position recorded: {max_pos}")
        
        # Calculate optimal settings
        range_size = max_pos - min_pos
        center_pos = (min_pos + max_pos) // 2
        
        print(f"Range: {range_size} units, Center: {center_pos}")
        
        # Ask user for preferences
        print(f"\nOptions for joint {joint_id}:")
        print(f"1. Set limits to [{min_pos}, {max_pos}] and offset to {center_pos}")
        print(f"2. Set limits to [{min_pos}, {max_pos}] and offset to {min_pos}")
        print(f"3. Reverse direction (phase=1) and adjust limits")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == "1":
            self.set_servo_limits(joint_id, min_pos, max_pos)
            self.set_servo_offset(joint_id, center_pos)
            phase = 0
        elif choice == "2":
            self.set_servo_limits(joint_id, min_pos, max_pos)
            self.set_servo_offset(joint_id, min_pos)
            phase = 0
        elif choice == "3":
            # Reverse direction
            self.set_servo_phase(joint_id, 1)
            # Adjust limits for reversed direction
            new_min = 4096 - max_pos
            new_max = 4096 - min_pos
            self.set_servo_limits(joint_id, new_min, new_max)
            self.set_servo_offset(joint_id, (new_min + new_max) // 2)
            phase = 1
        else:
            print("Invalid choice, skipping this joint")
            return None
        
        # Store calibration data
        self.calibration_data[joint_id] = {
            'min_pos': min_pos,
            'max_pos': max_pos,
            'phase': phase,
            'offset': self.read_position(joint_id) if choice == "1" else min_pos if choice == "2" else (new_min + new_max) // 2
        }
        
        print(f"Joint {joint_id} calibration complete!")
        return self.calibration_data[joint_id]
    
    def calibrate_all_joints(self):
        """Calibrate all joints"""
        print("=== Feetech Servo Calibration ===")
        print("This will help you set optimal limits and zero positions for each servo.")
        print("Make sure the robot is in a safe position before starting.\n")
        
        input("Press Enter when ready to begin calibration...")
        
        for joint_id in range(1, self.num_joints + 1):
            try:
                self.calibrate_joint(joint_id)
            except KeyboardInterrupt:
                print(f"\nCalibration interrupted at joint {joint_id}")
                break
            except Exception as e:
                print(f"Error calibrating joint {joint_id}: {e}")
                continue
        
        # Save calibration data
        self.save_calibration()
        
    def save_calibration(self):
        """Save calibration data to file"""
        filename = "feetech_calibration.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.calibration_data, f)
        print(f"\nCalibration data saved to {filename}")
        
        # Print summary
        print("\n=== Calibration Summary ===")
        for joint_id, data in self.calibration_data.items():
            print(f"Joint {joint_id}: min={data['min_pos']}, max={data['max_pos']}, "
                  f"phase={data['phase']}, offset={data['offset']}")
    
    def load_calibration(self, filename="feetech_calibration.pkl"):
        """Load calibration data from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.calibration_data = pickle.load(f)
            print(f"Loaded calibration data from {filename}")
            return True
        else:
            print(f"No calibration file found: {filename}")
            return False
    
    def apply_calibration(self):
        """Apply loaded calibration settings to servos"""
        if not self.calibration_data:
            print("No calibration data loaded!")
            return
        
        print("Applying calibration settings...")
        for joint_id, data in self.calibration_data.items():
            self.set_servo_limits(joint_id, data['min_pos'], data['max_pos'])
            self.set_servo_offset(joint_id, data['offset'])
            self.set_servo_phase(joint_id, data['phase'])
        
        print("Calibration settings applied!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Feetech Servo Calibration Tool")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--apply", action="store_true", help="Apply saved calibration")
    parser.add_argument("--file", default="feetech_calibration.pkl", help="Calibration file name")
    
    args = parser.parse_args()
    
    calib = FeetechCalibration()
    
    if args.calibrate:
        calib.calibrate_all_joints()
    elif args.apply:
        if calib.load_calibration(args.file):
            calib.apply_calibration()
    else:
        print("Use --calibrate to run calibration or --apply to apply saved calibration")

if __name__ == "__main__":
    main()
