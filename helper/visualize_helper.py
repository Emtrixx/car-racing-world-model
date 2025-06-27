import gymnasium as gym
import matplotlib.pyplot as plt

# Create the raw, unprocessed environment
env = gym.make('CarRacing-v3')
raw_observation, _ = env.reset()
# Skip the first 50 frames to leave out initial zooming in
for _ in range(50):
    action = env.action_space.sample()  # Random action to skip initial frames
    raw_observation, reward, terminated, truncated, info = env.step(action)
env.close()

cropped_observation = raw_observation[:-12, :, :]  # Crop the image to remove the top 84 pixels
# Display the raw observation
plt.imshow(cropped_observation)
plt.title("Raw 96x96 Observation")
# plt.axhline(y=83.5, color='r', linestyle='--', label='Crop line at y=84')
plt.legend()
plt.show()

print(f"Image shape is: {cropped_observation.shape}")
