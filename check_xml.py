import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("new_work2/shadow_hand/scene_right.xml")
data = mujoco.MjData(model)

# Set initial thumb position
joint_id_thj4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rh_THJ4")
if joint_id_thj4 != -1:
    data.qpos[joint_id_thj4] = 1.22173
else:
    print("Joint rh_THJ4 not found")

# Initialize data storage for force and torque
force_data = {finger: [] for finger in ['FF', 'MF', 'RF', 'LF', 'TH']}
torque_data = {finger: [] for finger in ['FF', 'MF', 'RF', 'LF', 'TH']}

# Get site IDs for deformation measurement
cyl_imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'cyl_imu')

# Get the cylinder's geom ID and initial properties
cylinder_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'soft_cylinder')
initial_radius = model.geom_size[cylinder_geom_id, 0]  # First element is radius for cylinder
initial_height = model.geom_size[cylinder_geom_id, 1] * 2  # Second element is half-height
print(f"Initial cylinder radius: {initial_radius}, height: {initial_height}")

# Step once to get initial positions and calculate initial volume
mujoco.mj_step(model, data)
cylinder_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'soft_cylinder')
initial_center_pos = data.site_xpos[cyl_imu_id].copy()

# Calculate initial volume of the cylinder
initial_volume = np.pi * (initial_radius**2) * initial_height
print(f"Initial volume of cylinder: {initial_volume} m³")

# Create empty lists for deformation data
deformation_data = []
ff_force_data = []
volume_data = []

# Get body IDs for contact detection
contact_bodies = {
    "rh_ffproximal": ('FF', 'J3'), "rh_ffmiddle": ('FF', 'J2'), "rh_ffdistal": ('FF', 'J1'),
    "rh_mfproximal": ('MF', 'J3'), "rh_mfmiddle": ('MF', 'J2'), "rh_mfdistal": ('MF', 'J1'),
    "rh_rfproximal": ('RF', 'J3'), "rh_rfmiddle": ('RF', 'J2'), "rh_rfdistal": ('RF', 'J1'),
    "rh_lfproximal": ('LF', 'J3'), "rh_lfmiddle": ('LF', 'J2'), "rh_lfdistal": ('LF', 'J1'),
    "rh_thmiddle": ('TH', 'J2'), "rh_thdistal": ('TH', 'J1')
}

# Get motor IDs for control
motor_names = [
    "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ2", "rh_THJ1", "rail_to_base"
]

motor_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in motor_names}

# Get the control ranges for the selected motors
motor_ctrlranges = {name: model.actuator_ctrlrange[motor_ids[name]] for name in motor_names}

# Initial torque is zero for all controlled motors
data.ctrl[[motor_ids[name] for name in motor_names]] = 0.0

# Rate of torque increase per simulation step
torque_increment_rate = 0.000005  # Very slow increase for precise control

# State tracking for contact detection
j3_contacted = {'FF': False, 'MF': False, 'RF': False, 'LF': False}
j2_contacted = {'FF': False, 'MF': False, 'RF': False, 'LF': False}
j1_contacted = {'FF': False, 'MF': False, 'RF': False, 'LF': False}
thj2_contacted = False
thj1_contacted = False

# Function to calculate signed volume of tetrahedron as described in section 4.3 of the paper
def calculate_signed_volume(point_a, point_b, point_c, origin):
    """
    Calculate the signed volume of a tetrahedron formed by three points and an origin
    Using the formula: Signed Volume = AO·(AB×AC)/6
    """
    ao = point_a - origin
    ab = point_b - point_a
    ac = point_c - point_a
    
    # Calculate the cross product AB × AC
    cross_product = np.cross(ab, ac)
    
    # Calculate the dot product AO·(AB×AC)
    dot_product = np.dot(ao, cross_product)
    
    # Return the signed volume (divided by 6)
    return dot_product / 6.0

# Function to estimate cylinder deformation using contact points
def estimate_cylinder_deformation(contact_pos, cylinder_center, initial_radius):
    """
    Estimate the deformation of the cylinder based on contact position
    """
    # Vector from center to contact point
    center_to_contact = contact_pos - cylinder_center
    
    # Project onto horizontal plane (ignore z-component)
    horizontal_vector = np.array([center_to_contact[0], center_to_contact[1], 0])
    horizontal_distance = np.linalg.norm(horizontal_vector)
    
    # Calculate deformation (positive value means compression)
    deformation = initial_radius - horizontal_distance
    
    return deformation

# For visualization
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Simulation loop
    while viewer.is_running():
        # Modified control strategy: Apply torque to all joints simultaneously
        # but with different rates to ensure proper wrapping
        for finger in ['FF', 'MF', 'RF', 'LF']:
            # Apply torque to J3 (proximal) with slower rate
            if not j3_contacted[finger]:
                motor_name = f"rh_{finger}J3"
                motor_id = motor_ids[motor_name]
                ctrl_range = motor_ctrlranges[motor_name]
                # Use 60% of max torque for J3 to allow more natural wrapping
                target_torque = ctrl_range[1] * 0.6
                data.ctrl[motor_id] = min(data.ctrl[motor_id] + torque_increment_rate * 0.5, target_torque)

            # Apply torque to J2 (middle) with higher priority
            if not j2_contacted[finger]:
                motor_name = f"rh_{finger}J2"
                motor_id = motor_ids[motor_name]
                ctrl_range = motor_ctrlranges[motor_name]
                # Use 80% of max torque for better control of middle joint
                target_torque = ctrl_range[1] * 0.8
                data.ctrl[motor_id] = min(data.ctrl[motor_id] + torque_increment_rate * 1.0, target_torque)

            # Apply torque to J1 (distal/fingertip) with highest priority
            if not j1_contacted[finger]:
                motor_name = f"rh_{finger}J1"
                motor_id = motor_ids[motor_name]
                ctrl_range = motor_ctrlranges[motor_name]
                # Use 90% of max torque for fingertip to ensure good contact
                target_torque = ctrl_range[1] * 0.9
                data.ctrl[motor_id] = min(data.ctrl[motor_id] + torque_increment_rate * 1.5, target_torque)

        # Control Thumb (THJ2 then THJ1) with higher torque for opposition
        if not thj2_contacted:
            motor_name = "rh_THJ2"
            motor_id = motor_ids[motor_name]
            ctrl_range = motor_ctrlranges[motor_name]
            target_torque = ctrl_range[1] * 0.9
            data.ctrl[motor_id] = min(data.ctrl[motor_id] + torque_increment_rate * 1.2, target_torque)

        if not thj1_contacted:
            motor_name = "rh_THJ1"
            motor_id = motor_ids[motor_name]
            ctrl_range = motor_ctrlranges[motor_name]
            target_torque = ctrl_range[1] * 0.9
            data.ctrl[motor_id] = min(data.ctrl[motor_id] + torque_increment_rate * 1.5, target_torque)

        ff_contact_pos = None
        ff_contact_force = 0
        contact_points = []

        # Check for contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # Get body IDs from geom IDs
            body1_id = model.geom_bodyid[geom1_id]
            body2_id = model.geom_bodyid[geom2_id]
            
            # Get body names
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            
            # Check if contact is between a hand/thumb body and the cylinder
            hand_body_name = None
            if body1_name in contact_bodies and body2_name == "soft_cylinder":
                hand_body_name = body1_name
                # Store contact point for deformation calculation
                contact_points.append(contact.pos.copy())
            elif body2_name in contact_bodies and body1_name == "soft_cylinder":
                hand_body_name = body2_name
                # Store contact point for deformation calculation
                contact_points.append(contact.pos.copy())
            
            if hand_body_name:
                finger_name, joint_type = contact_bodies[hand_body_name]
                
                # Get contact force using mj_contactForce
                c_array = np.zeros(6, dtype=np.float64)
                mujoco._functions.mj_contactForce(model, data, i, c_array)
                contact_force = np.linalg.norm(c_array[:3])
                
                # Record force and torque for plotting
                if joint_type == 'J1':  # Fingertip
                    torque = data.ctrl[motor_ids[f"rh_{finger_name}J1"]]
                    force_data[finger_name].append(contact_force)
                    torque_data[finger_name].append(torque)
                    
                    # For FF specifically, record position for deformation
                    if finger_name == 'FF':
                        ff_contact_pos = contact.pos.copy()
                        ff_contact_force = contact_force
                
                # Process finger contacts - immediately stop torque on contact
                if finger_name != 'TH':  # Handle fingers
                    if joint_type == 'J3' and not j3_contacted[finger_name]:
                        print(f"Contact detected with {hand_body_name} (J3 of {finger_name}). Force: {contact_force:.2f}. Setting torque to zero.")
                        j3_contacted[finger_name] = True
                        data.ctrl[motor_ids[f"rh_{finger_name}J3"]] = 0.0
                    
                    elif joint_type == 'J2' and not j2_contacted[finger_name]:
                        print(f"Contact detected with {hand_body_name} (J2 of {finger_name}). Force: {contact_force:.2f}. Setting torque to zero.")
                        j2_contacted[finger_name] = True
                        j3_contacted[finger_name] = True
                        data.ctrl[motor_ids[f"rh_{finger_name}J2"]] = 0.0
                        data.ctrl[motor_ids[f"rh_{finger_name}J3"]] = 0.0
                    
                    elif joint_type == 'J1' and not j1_contacted[finger_name]:
                        print(f"Contact detected with {hand_body_name} (J1 of {finger_name}). Force: {contact_force:.2f}. Setting torque to zero.")
                        j1_contacted[finger_name] = True
                        data.ctrl[motor_ids[f"rh_{finger_name}J1"]] = 0.0
                        j2_contacted[finger_name] = True
                        j3_contacted[finger_name] = True
                        data.ctrl[motor_ids[f"rh_{finger_name}J2"]] = 0.0
                        data.ctrl[motor_ids[f"rh_{finger_name}J3"]] = 0.0
                
                # Process thumb contacts
                else:  # Handle thumb
                    if joint_type == 'J2' and not thj2_contacted:
                        print(f"Contact detected with {hand_body_name} (THJ2). Force: {contact_force:.2f}. Setting torque to zero.")
                        thj2_contacted = True
                        data.ctrl[motor_ids["rh_THJ2"]] = 0.0
                    
                    elif joint_type == 'J1' and not thj1_contacted:
                        print(f"Contact detected with {hand_body_name} (THJ1). Force: {contact_force:.2f}. Setting torque to zero.")
                        thj1_contacted = True
                        data.ctrl[motor_ids["rh_THJ1"]] = 0.0
                        thj2_contacted = True
                        data.ctrl[motor_ids["rh_THJ2"]] = 0.0

        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Get current center position
        current_center_pos = data.site_xpos[cyl_imu_id]
        
        # Calculate deformation using signed volume method if we have contact points
        if len(contact_points) >= 3:
            # Use the contact points to form tetrahedra with the cylinder center
            total_volume = 0
            
            # Calculate volume using the method from section 4.3 of the paper
            # "To estimate volume changes for deformable objects devoid of explicit mathematical formulations, 
            # a general point (shifted origin) is defined to construct a tetrahedron"
            for i in range(len(contact_points) - 2):
                vol = calculate_signed_volume(
                    contact_points[i],
                    contact_points[i+1],
                    contact_points[i+2],
                    current_center_pos
                )
                total_volume += abs(vol)  # Use absolute value as we care about magnitude
            
            # Calculate deformation as change in volume
            volume_deformation = initial_volume - total_volume
            volume_data.append(volume_deformation)
            
            # If FF finger is in contact, calculate horizontal deformation
            if ff_contact_pos is not None:
                horizontal_deformation = estimate_cylinder_deformation(ff_contact_pos, current_center_pos, initial_radius)
                deformation_data.append(horizontal_deformation)
                ff_force_data.append(ff_contact_force)
                
                if len(deformation_data) % 100 == 0:
                    print(f"FF Force: {ff_contact_force:.4f}, Horizontal deformation: {horizontal_deformation:.6f}")
                    print(f"Volume deformation: {volume_deformation:.6f}")
            else:
                # No FF contact, use previous values or zeros
                if len(deformation_data) > 0:
                    deformation_data.append(deformation_data[-1])
                else:
                    deformation_data.append(0)
                
                if len(ff_force_data) > 0:
                    ff_force_data.append(0)
                else:
                    ff_force_data.append(0)
        else:
            # Not enough contact points for volume calculation
            if len(volume_data) > 0:
                volume_data.append(volume_data[-1])
            else:
                volume_data.append(0)
            
            # Calculate horizontal deformation if FF is in contact
            if ff_contact_pos is not None:
                horizontal_deformation = estimate_cylinder_deformation(ff_contact_pos, current_center_pos, initial_radius)
                deformation_data.append(horizontal_deformation)
                ff_force_data.append(ff_contact_force)
            else:
                # No FF contact, use previous values or zeros
                if len(deformation_data) > 0:
                    deformation_data.append(deformation_data[-1])
                else:
                    deformation_data.append(0)
                
                if len(ff_force_data) > 0:
                    ff_force_data.append(0)
                else:
                    ff_force_data.append(0)
        
        # Sync the viewer
        viewer.sync()

    # Plot fingertip forces vs. torque for all fingers
    plt.figure(figsize=(10, 6))
    for finger in ['FF', 'MF', 'RF', 'LF', 'TH']:
        if torque_data[finger]:  # Only plot if data exists
            plt.plot(torque_data[finger], force_data[finger], 'o-', label=finger)
    
    plt.xlabel("Applied Torque (Nm)", fontsize=12)
    plt.ylabel("Contact Force at Fingertip (N)", fontsize=12)
    plt.title("Fingertip Force vs. Applied Torque", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fingertip_force_vs_torque.png', dpi=300)
    plt.show()

    # Create a new list that pairs FF forces with horizontal deformations
    ff_force_deform = []
    for i in range(min(len(ff_force_data), len(deformation_data))):
        if ff_force_data[i] > 0.01:  # Only include meaningful forces
            ff_force_deform.append((ff_force_data[i], deformation_data[i]))
    
    # Sort by force to ensure a clean plot
    ff_force_deform.sort()
    
    # Extract sorted forces and deformations
    ff_forces = [x[0] for x in ff_force_deform]
    deformations = [x[1] for x in ff_force_deform]
    
    # Plot horizontal deformation vs. FF force
    plt.figure(figsize=(10, 6))
    plt.plot(ff_forces, deformations, 'o-', color='blue', linewidth=2)
    plt.xlabel("Force on FF (N)", fontsize=12)
    plt.ylabel("Horizontal Deformation (m)", fontsize=12)
    plt.title("Cylinder Horizontal Deformation vs. Force on First Finger", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('ff_force_vs_horizontal_deformation.png', dpi=300)
    plt.show()

    # Plot volume deformation vs. FF force if we have volume data
    if len(volume_data) > 0:
        # Create a new list that pairs FF forces with volume deformations
        ff_force_vol_deform = []
        for i in range(min(len(ff_force_data), len(volume_data))):
            if ff_force_data[i] > 0.01:  # Only include meaningful forces
                ff_force_vol_deform.append((ff_force_data[i], volume_data[i]))
        
        # Sort by force to ensure a clean plot
        ff_force_vol_deform.sort()
        
        # Extract sorted forces and volume deformations
        ff_forces_vol = [x[0] for x in ff_force_vol_deform]
        vol_deformations = [x[1] for x in ff_force_vol_deform]
        
        # Plot volume deformation vs. FF force
        plt.figure(figsize=(10, 6))
        plt.plot(ff_forces_vol, vol_deformations, 'o-', color='red', linewidth=2)
        plt.xlabel("Force on FF (N)", fontsize=12)
        plt.ylabel("Volume Deformation (m³)", fontsize=12)
        plt.title("Cylinder Volume Deformation vs. Force on First Finger", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('ff_force_vs_volume_deformation.png', dpi=300)
        plt.show()

    print("Simulation finished.")
    
#     Now I want to do the similar thing with the environment we created (env_test.py) in mujoco. 
# The changes I want to incorporate are:
# 1. slip detection should be done using camera feed (using the frame difference method we can check if there is slip or not, add a camera on the wrist so that it moves with the hand and when it will grasp the object and lift it, we can check whether the object is slipping or not).
# 2. Deformation will still be calculated using the signed volume method.
# 3. The state space/observation space will consist of 14 joint angles + 14 joint velocities (rh_THJ1, rh_THJ2, rh_FFJ1, rh_FFJ2, rh_FFJ3, rh_MFJ1, rh_MFJ2, rh_MFJ3, rh_RFJ1, rh_RFJ2, rh_RFJ3, rh_LFJ1, rh_LFJ2, rh_LFJ3) + 5 finger tip forces.
# 4. The action space will consist of 14 joint torques.