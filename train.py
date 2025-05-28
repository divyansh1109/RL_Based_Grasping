import os
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from soft_object_grasp_env import SoftObjectGraspEnv

# Create directories for models and logs
models_dir = "models/SAC"
logs_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Custom callback for printing training progress
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_mean_reward = -np.inf
        self.successful_lifts = 0
        self.total_episodes = 0
    
    def _on_step(self):
        # Check if episode has ended
        if self.locals.get("dones")[0]:
            self.episode_count += 1
            self.total_episodes += 1
            reward = sum(self.locals.get("rewards"))
            self.episode_rewards.append(reward)
            
            # Check if lift was successful
            info = self.locals.get("infos")[0]
            if info.get("successful_lift", False):
                self.successful_lifts += 1
            
            # Print episode stats every 10 episodes
            if self.episode_count % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                success_rate = self.successful_lifts / max(1, self.total_episodes) * 100
                print(f"Episode: {self.total_episodes}, Mean Reward: {mean_reward:.2f}, Success Rate: {success_rate:.2f}%")
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(f"{models_dir}/best_model")
                    print(f"New best model saved with mean reward: {mean_reward:.2f}")
                
                # Reset stats
                self.episode_rewards = []
                self.episode_count = 0
        
        return True

# Create and wrap the environment
def make_env():
    env = SoftObjectGraspEnv()
    env = Monitor(env)
    return env

# Create vectorized environment
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Create SAC agent with optimized hyperparameters
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    ent_coef='auto',
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    learning_starts=1000,
    use_sde=False,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256],
            qf=[256, 256]
        )
    ),
    verbose=1,
    tensorboard_log=logs_dir
)

# Set up callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=models_dir,
    name_prefix="sac_grasping"
)

training_callback = TrainingCallback()

# Train the agent
total_timesteps = 1000000
print(f"Starting training for {total_timesteps} timesteps...")
start_time = time.time()

model.learn(
    total_timesteps=total_timesteps,
    callback=[checkpoint_callback, training_callback],
    tb_log_name="SAC_grasping"
)

# Save the final model
model.save(f"{models_dir}/final_model")
env.save(f"{models_dir}/vec_normalize.pkl")

# Calculate training time
training_time = time.time() - start_time
hours, remainder = divmod(training_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Evaluate the trained agent
print("Evaluating trained agent...")
eval_env = make_env()
eval_env = VecNormalize.load(f"{models_dir}/vec_normalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()
