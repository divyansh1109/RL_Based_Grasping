import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

class SoftObjectSimulation:
    def __init__(self, xml_path):
        """Initialize the simulation with the given XML model file."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
    
        self.joint_names = []
        self.joint_velocities = {}
        
        for finger in ['FF', 'MF', 'RF', 'LF', 'TH']:
            for joint_num in ['1', '2', '3', '4']:
                joint_name = f"rh_{finger}J{joint_num}"
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id != -1:  
                    self.joint_names.append(joint_name)
                    self.joint_velocities[joint_name] = []
        
        self.force_data = {finger: [] for finger in ['FF', 'MF', 'RF', 'LF', 'TH']}
        self.torque_data = {finger: [] for finger in ['FF', 'MF', 'RF', 'LF', 'TH']}
        self.deformation_data = []
        self.ff_force_data = []
        self.volume_data = []
        
        self.cyl_imu_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'cyl_imu')
        self.cylinder_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'soft_cylinder')
        
        self.initial_radius = self.model.geom_size[self.cylinder_geom_id, 0]
        self.initial_height = self.model.geom_size[self.cylinder_geom_id, 1] * 2
        print(f"Initial cylinder radius: {self.initial_radius}, height: {self.initial_height}")
        
        mujoco.mj_step(self.model, self.data)
        self.cylinder_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'soft_cylinder')
        self.initial_center_pos = self.data.site_xpos[self.cyl_imu_id].copy()
        
        self.initial_volume = np.pi * (self.initial_radius**2) * self.initial_height
        print(f"Initial volume of cylinder: {self.initial_volume} m³")
        
        self.contact_bodies = {
            "rh_ffproximal": ('FF', 'J3'), "rh_ffmiddle": ('FF', 'J2'), "rh_ffdistal": ('FF', 'J1'),
            "rh_mfproximal": ('MF', 'J3'), "rh_mfmiddle": ('MF', 'J2'), "rh_mfdistal": ('MF', 'J1'),
            "rh_rfproximal": ('RF', 'J3'), "rh_rfmiddle": ('RF', 'J2'), "rh_rfdistal": ('RF', 'J1'),
            "rh_lfproximal": ('LF', 'J3'), "rh_lfmiddle": ('LF', 'J2'), "rh_lfdistal": ('LF', 'J1'),
            "rh_thmiddle": ('TH', 'J2'), "rh_thdistal": ('TH', 'J1')
        }
        
        self.motor_names = [
            "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
            "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
            "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
            "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
            "rh_THJ2", "rh_THJ1", "rail_to_base"
        ]
        
        self.motor_ids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                          for name in self.motor_names}
        self.motor_ctrlranges = {name: self.model.actuator_ctrlrange[self.motor_ids[name]] 
                                for name in self.motor_names}
        
        self.data.ctrl[[self.motor_ids[name] for name in self.motor_names]] = 0.0
        
        self.torque_increment_rate = 0.000005
        
        self.j3_contacted = {'FF': False, 'MF': False, 'RF': False, 'LF': False}
        self.j2_contacted = {'FF': False, 'MF': False, 'RF': False, 'LF': False}
        self.j1_contacted = {'FF': False, 'MF': False, 'RF': False, 'LF': False}
        self.thj2_contacted = False
        self.thj1_contacted = False
        
        self.ff_contact_pos = None
        self.ff_contact_force = 0
        self.contact_points = []
        
    def print_joint_info(self):
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"Joint {i}: {name}")

        
    def calculate_joint_velocities(self):
        """Calculate and store velocities for all tracked joints"""
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                joint_vel = self.data.qvel[joint_id]
                self.joint_velocities[joint_name].append(joint_vel)
                

    def calculate_signed_volume(self, point_a, point_b, point_c, origin):
        """
        Calculate the signed volume of a tetrahedron formed by three points and an origin
        Using the formula: Signed Volume = AO·(AB×AC)/6
        """
        ao = point_a - origin
        ab = point_b - point_a
        ac = point_c - point_a
        
        cross_product = np.cross(ab, ac)
        
        dot_product = np.dot(ao, cross_product)
        
        return dot_product / 6.0

    def estimate_cylinder_deformation(self, contact_pos, cylinder_center, initial_radius):
        """
        Estimate the deformation of the cylinder based on contact position
        """
        center_to_contact = contact_pos - cylinder_center
        
        horizontal_vector = np.array([center_to_contact[0], center_to_contact[1], 0])
        horizontal_distance = np.linalg.norm(horizontal_vector)
        
        deformation = initial_radius - horizontal_distance
        
        return deformation

    def control_torques(self):
        """Apply torques to all finger joints based on contact state"""
        for finger in ['FF', 'MF', 'RF', 'LF']:
            if not self.j3_contacted[finger]:
                motor_name = f"rh_{finger}J3"
                motor_id = self.motor_ids[motor_name]
                ctrl_range = self.motor_ctrlranges[motor_name]
                target_torque = ctrl_range[1] * 0.6
                self.data.ctrl[motor_id] = min(self.data.ctrl[motor_id] + self.torque_increment_rate * 0.5, target_torque)

            if not self.j2_contacted[finger]:
                motor_name = f"rh_{finger}J2"
                motor_id = self.motor_ids[motor_name]
                ctrl_range = self.motor_ctrlranges[motor_name]
                target_torque = ctrl_range[1] * 0.8
                self.data.ctrl[motor_id] = min(self.data.ctrl[motor_id] + self.torque_increment_rate * 1.0, target_torque)

            if not self.j1_contacted[finger]:
                motor_name = f"rh_{finger}J1"
                motor_id = self.motor_ids[motor_name]
                ctrl_range = self.motor_ctrlranges[motor_name]
                target_torque = ctrl_range[1] * 0.9
                self.data.ctrl[motor_id] = min(self.data.ctrl[motor_id] + self.torque_increment_rate * 1.5, target_torque)

        if not self.thj2_contacted:
            motor_name = "rh_THJ2"
            motor_id = self.motor_ids[motor_name]
            ctrl_range = self.motor_ctrlranges[motor_name]
            target_torque = ctrl_range[1] * 0.9
            self.data.ctrl[motor_id] = min(self.data.ctrl[motor_id] + self.torque_increment_rate * 1.2, target_torque)

        if not self.thj1_contacted:
            motor_name = "rh_THJ1"
            motor_id = self.motor_ids[motor_name]
            ctrl_range = self.motor_ctrlranges[motor_name]
            target_torque = ctrl_range[1] * 0.9
            self.data.ctrl[motor_id] = min(self.data.ctrl[motor_id] + self.torque_increment_rate * 1.5, target_torque)

    def detect_contacts(self):
        """Detect contacts between fingers and cylinder, update contact state"""
        self.ff_contact_pos = None
        self.ff_contact_force = 0
        self.contact_points = []
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            
            hand_body_name = None
            if body1_name in self.contact_bodies and body2_name == "soft_cylinder":
                hand_body_name = body1_name
                self.contact_points.append(contact.pos.copy())
            elif body2_name in self.contact_bodies and body1_name == "soft_cylinder":
                hand_body_name = body2_name
                self.contact_points.append(contact.pos.copy())
            
            if hand_body_name:
                finger_name, joint_type = self.contact_bodies[hand_body_name]
                
                c_array = np.zeros(6, dtype=np.float64)
                mujoco._functions.mj_contactForce(self.model, self.data, i, c_array)
                contact_force = np.linalg.norm(c_array[:3])
                
                if joint_type == 'J1':  # Fingertip
                    torque = self.data.ctrl[self.motor_ids[f"rh_{finger_name}J1"]]
                    self.force_data[finger_name].append(contact_force)
                    self.torque_data[finger_name].append(torque)
                    
                    if finger_name == 'FF':
                        self.ff_contact_pos = contact.pos.copy()
                        self.ff_contact_force = contact_force
                                        
                self.update_contact_state(finger_name, joint_type, hand_body_name, contact_force)

    def update_contact_state(self, finger_name, joint_type, hand_body_name, contact_force):
        """Update contact state and adjust torques based on detected contacts"""
        if finger_name != 'TH':  
            if joint_type == 'J3' and not self.j3_contacted[finger_name]:
                print(f"Contact detected with {hand_body_name} (J3 of {finger_name}). Force: {contact_force:.2f}. Setting torque to zero.")
                self.j3_contacted[finger_name] = True
                self.data.ctrl[self.motor_ids[f"rh_{finger_name}J3"]] = 0.0
            
            elif joint_type == 'J2' and not self.j2_contacted[finger_name]:
                print(f"Contact detected with {hand_body_name} (J2 of {finger_name}). Force: {contact_force:.2f}. Setting torque to zero.")
                self.j2_contacted[finger_name] = True
                self.j3_contacted[finger_name] = True
                self.data.ctrl[self.motor_ids[f"rh_{finger_name}J2"]] = 0.0
                self.data.ctrl[self.motor_ids[f"rh_{finger_name}J3"]] = 0.0
                    
            elif joint_type == 'J1' and not self.j1_contacted[finger_name]:
                print(f"Contact detected with {hand_body_name} (J1 of {finger_name}). Force: {contact_force:.2f}. Setting torque to zero.")
                self.j1_contacted[finger_name] = True
                self.data.ctrl[self.motor_ids[f"rh_{finger_name}J1"]] = 0.0
                self.j2_contacted[finger_name] = True
                self.j3_contacted[finger_name] = True
                self.data.ctrl[self.motor_ids[f"rh_{finger_name}J2"]] = 0.0
                self.data.ctrl[self.motor_ids[f"rh_{finger_name}J3"]] = 0.0
        
        else:  
            if joint_type == 'J2' and not self.thj2_contacted:
                print(f"Contact detected with {hand_body_name} (THJ2). Force: {contact_force:.2f}. Setting torque to zero.")
                self.thj2_contacted = True
                self.data.ctrl[self.motor_ids["rh_THJ2"]] = 0.0
            
            elif joint_type == 'J1' and not self.thj1_contacted:
                print(f"Contact detected with {hand_body_name} (THJ1). Force: {contact_force:.2f}. Setting torque to zero.")
                self.thj1_contacted = True
                self.data.ctrl[self.motor_ids["rh_THJ1"]] = 0.0
                self.thj2_contacted = True
                self.data.ctrl[self.motor_ids["rh_THJ2"]] = 0.0
                
        self.data.ctrl[self.motor_ids[f"rail_to_base"]] = -2000

    def calculate_volume_deformation(self):
        """Calculate volume deformation using signed volume method"""
        current_center_pos = self.data.site_xpos[self.cyl_imu_id]
        
        if len(self.contact_points) >= 3:
            total_volume = 0
            
            for i in range(len(self.contact_points) - 2):
                vol = self.calculate_signed_volume(
                    self.contact_points[i],
                    self.contact_points[i+1],
                    self.contact_points[i+2],
                    current_center_pos
                )
                total_volume += abs(vol)
            
            volume_deformation = self.initial_volume - total_volume
            self.volume_data.append(volume_deformation)
            return volume_deformation
        else:
            if len(self.volume_data) > 0:
                self.volume_data.append(self.volume_data[-1])
            else:
                self.volume_data.append(0)
            return 0

    def calculate_horizontal_deformation(self):
        """Calculate horizontal deformation based on FF contact"""
        current_center_pos = self.data.site_xpos[self.cyl_imu_id]
        
        if self.ff_contact_pos is not None:
            horizontal_deformation = self.estimate_cylinder_deformation(
                self.ff_contact_pos, current_center_pos, self.initial_radius
            )
            self.deformation_data.append(horizontal_deformation)
            self.ff_force_data.append(self.ff_contact_force)
        
            return horizontal_deformation
        else:
            if len(self.deformation_data) > 0:
                self.deformation_data.append(self.deformation_data[-1])
            else:
                self.deformation_data.append(0)
            
            if len(self.ff_force_data) > 0:
                self.ff_force_data.append(0)
            else:
                self.ff_force_data.append(0)
            
            return 0

    def step(self):
        """Execute one simulation step: control, detect contacts, calculate deformation"""
        self.control_torques()
        
        self.detect_contacts()
        
        mujoco.mj_step(self.model, self.data)
        
        volume_deformation = self.calculate_volume_deformation()
        horizontal_deformation = self.calculate_horizontal_deformation()
        
        self.calculate_joint_velocities()
        
        return horizontal_deformation, volume_deformation
    
    def plot_joint_velocities(self):
        """Plot joint velocities over time"""
        plt.figure(figsize=(12, 8))
        
        fingers = ['FF', 'MF', 'RF', 'LF', 'TH']
        for i, finger in enumerate(fingers):
            plt.subplot(3, 2, i+1)
            
            for joint_num in ['1', '2', '3', '4']:
                joint_name = f"rh_{finger}J{joint_num}"
                if joint_name in self.joint_velocities and self.joint_velocities[joint_name]:
                    plt.plot(self.joint_velocities[joint_name], label=f"J{joint_num}")
            
            plt.title(f"{finger} Joint Velocities")
            plt.xlabel("Simulation Steps")
            plt.ylabel("Velocity (rad/s)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
        
        plt.tight_layout()
        plt.show()


    def plot_force_vs_torque(self):
        """Plot fingertip forces vs. torque for all fingers"""
        plt.figure(figsize=(10, 6))
        for finger in ['FF', 'MF', 'RF', 'LF', 'TH']:
            if self.torque_data[finger]:  
                plt.plot(self.torque_data[finger], self.force_data[finger], 'o-', label=finger)
        
        plt.xlabel("Applied Torque (Nm)", fontsize=12)
        plt.ylabel("Contact Force at Fingertip (N)", fontsize=12)
        plt.title("Fingertip Force vs. Applied Torque", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def plot_volume_deformation_vs_force(self):
        """Plot volume deformation vs. FF force"""
            
        ff_force_vol_deform = []
        for i in range(min(len(self.ff_force_data), len(self.volume_data))):
            if self.ff_force_data[i] > 0.01:  
                ff_force_vol_deform.append((self.ff_force_data[i], self.volume_data[i]))
            
        ff_force_vol_deform.sort()
        
        ff_forces_vol = [x[0] for x in ff_force_vol_deform]
        vol_deformations = [x[1] for x in ff_force_vol_deform]
        
        plt.figure(figsize=(10, 6))
        plt.plot(ff_forces_vol, vol_deformations, 'o-', color='red', linewidth=2)
        plt.xlabel("Force on FF (N)", fontsize=12)
        plt.ylabel("Volume Deformation (m³)", fontsize=12)
        plt.title("Cylinder Volume Deformation vs. Force on First Finger", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def run_simulation(self):
        """Run the main simulation loop"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                joint_id_thj4 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_THJ4")
                if joint_id_thj4 != -1:
                    self.data.qpos[joint_id_thj4] = 1.22173
                else:
                    print("Joint rh_THJ4 not found")
                self.step()
                
                viewer.sync()
            
            self.plot_force_vs_torque()
            self.plot_volume_deformation_vs_force()
            self.plot_joint_velocities()  
            
            print("Simulation finished.")


if __name__ == "__main__":
    simulation = SoftObjectSimulation("Soft_Object_Grasp/shadow_hand/scene_right.xml")
    simulation.run_simulation()