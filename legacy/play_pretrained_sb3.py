from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
import gymnasium as gym

# model_id = "Rinnnt/ppo-CarRacing-v3"
# model_filename = "ppo-CarRacing-v3.zip"
model_id = "Pyro-X2/CarRacingSB3"
model_filename = "ppo-CarRacing-v3.zip"

try:
    checkpoint = load_from_hub(model_id, model_filename)  # Load the model from Hugging Face Hub
except Exception as e:
    print(f"Failed to load model from Hugging Face Hub: {e}")
    exit(1)

# Create the environment
env = gym.make("CarRacing-v3", render_mode="human")
# Load the model
model = PPO.load(checkpoint, env=env)

# Reset the environment
observation, info = env.reset()
# Play the game using the loaded model
done = False
while not done:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
