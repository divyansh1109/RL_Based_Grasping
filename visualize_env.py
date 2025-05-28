import gymnasium as gym
import numpy as np
from soft_object_grasp_env import SoftObjectGraspEnv  # Import your custom environment

def run_random_simulation(num_episodes=5, max_steps=1000):
    # Create environment with human rendering
    env = SoftObjectGraspEnv(
        xml_path="Soft_Object_Grasp/shadow_hand/scene_right.xml",
        render_mode="human"
    )
    
    for episode in range(num_episodes):
        # Reset environment with proper initialization
        obs, _ = env.reset()
        
        # Allow time for physics to stabilize
        for _ in range(10):
            env.step(env.action_space.sample() * 0)  # Zero action
            env.render()
        
        terminated = False
        truncated = False
        step_count = 0
        
        print(f"\n=== Episode {episode+1}/{num_episodes} ===")
        print("Initial Observation:", obs.shape)
        
        while not (terminated or truncated) and step_count < max_steps:
            # Generate random action with higher torque values
            action = env.action_space.sample()
            # Scale finger joint torques to ensure movement
            # action[:-1] = action[:-1]  # Scale finger joint torques
            
            # Take a step with action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the environment
            env.render()
            
            # Print step information
            if step_count % 10 == 0:  # Print less frequently
                print(f"Step {step_count}:")
                print(f"Reward: {reward:.2f}")
                print(f"Deformation: {info['deformation']:.6f}")
                print(f"Slip Detected: {info['slip_detected']}")
                print(f"Phase: {info['phase']}")
                print("-" * 40)
            
            step_count += 1
            
            # Check early termination conditions
            if terminated or truncated:
                print("\nEpisode terminated early!")
                reason = "Unknown"
                if info.get('successful_lift', False):
                    reason = "Successful lift"
                else:
                    reason = "Timeout"
                print(f"Reason: {reason}")
                break
    
    env.close()


if __name__ == "__main__":
    run_random_simulation(num_episodes=3, max_steps=500)