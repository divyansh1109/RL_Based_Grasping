import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from soft_object_grasp_env import SoftObjectGraspEnv

# Path to the trained model
model_path = "models/SAC/final_model"
vec_normalize_path = "models/SAC/vec_normalize.pkl"

# Create environment
env = SoftObjectGraspEnv(render_mode="human")
env = DummyVecEnv([lambda: env])
env = VecNormalize.load(vec_normalize_path, env)
env.training = False
env.norm_reward = False

# Load the trained model
model = SAC.load(model_path)

# Lists to store data for plotting
rewards = []
deformations = []
slips = []
lift_heights = []
forces = []
phases = []

# Run episodes
n_episodes = 5
for episode in range(n_episodes):
    print(f"Episode {episode+1}/{n_episodes}")
    obs = env.reset()[0]
    episode_reward = 0
    episode_deformations = []
    episode_slips = []
    episode_heights = []
    episode_forces = []
    episode_phases = []
    
    done = False
    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated[0] or truncated[0]
        info = info[0]  # Unwrap info from vectorized env
        
        # Store data
        episode_reward += reward[0]
        episode_deformations.append(info['deformation'])
        episode_slips.append(int(info['slip_detected']))
        episode_heights.append(info.get('lift_height', 0))
        episode_forces.append(np.mean(info['fingertip_forces']))
        episode_phases.append(info['phase'])
    
    # Store episode data
    rewards.append(episode_reward)
    deformations.append(np.mean(episode_deformations))
    slips.append(np.sum(episode_slips))
    lift_heights.append(np.max(episode_heights))
    forces.append(np.mean(episode_forces))
    phases.append("success" if info.get('successful_lift', False) else "failure")
    
    print(f"Episode reward: {episode_reward:.2f}")
    print(f"Mean deformation: {np.mean(episode_deformations):.6f}")
    print(f"Number of slips: {np.sum(episode_slips)}")
    print(f"Max lift height: {np.max(episode_heights):.2f}")
    print(f"Mean force: {np.mean(episode_forces):.2f}")
    print(f"Final phase: {info['phase']}")
    print(f"Success: {info.get('successful_lift', False)}")
    print("-" * 50)

# Close environment
env.close()

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.bar(range(n_episodes), rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(2, 3, 2)
plt.bar(range(n_episodes), deformations)
plt.title('Mean Deformation')
plt.xlabel('Episode')
plt.ylabel('Deformation (m³)')

plt.subplot(2, 3, 3)
plt.bar(range(n_episodes), slips)
plt.title('Number of Slips')
plt.xlabel('Episode')
plt.ylabel('Slips')

plt.subplot(2, 3, 4)
plt.bar(range(n_episodes), lift_heights)
plt.title('Max Lift Height')
plt.xlabel('Episode')
plt.ylabel('Height (m)')

plt.subplot(2, 3, 5)
plt.bar(range(n_episodes), forces)
plt.title('Mean Force')
plt.xlabel('Episode')
plt.ylabel('Force (N)')

plt.subplot(2, 3, 6)
colors = ['green' if p == "success" else 'red' for p in phases]
plt.bar(range(n_episodes), [1]*n_episodes, color=colors)
plt.title('Success/Failure')
plt.xlabel('Episode')
plt.yticks([])

plt.tight_layout()
plt.savefig('test_results.png')
plt.show()
