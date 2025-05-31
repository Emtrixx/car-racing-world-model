# Reinforcement Learning with World Models for Car Racing

This project implements and trains agents for the CarRacing-v3 environment using reinforcement learning techniques, including Variational Autoencoders (VAEs), World Models (WM), and Proximal Policy Optimization (PPO).

## Project Overview

The primary goal is to train an agent that can drive effectively in the CarRacing-v3 environment. This involves several stages:

1.  **Visual Encoding:** A Convolutional Variational Autoencoder (ConvVAE) is trained to compress raw pixel observations from the environment into a lower-dimensional latent space.
2.  **World Model Training:** A Recurrent Neural Network (RNN-based, specifically GRU) World Model is trained to predict future latent states, rewards, and done flags based on current latent states and actions. This allows the agent to "dream" or simulate future trajectories.
3.  **Policy Training:** A PPO agent is trained using the learned World Model to make decisions. It can be trained either by interacting directly with the environment or by leveraging the World Model to generate simulated experiences (training in dream).

## Key Scripts

-   `train_vae.py`: Trains the Convolutional VAE to encode environment observations.
-   `train_world_model.py`: Trains the GRU-based World Model on sequences of latent states, actions, rewards, and done flags collected from the environment.
-   `train_world_model_parallel.py`: A version of `train_world_model.py` that utilizes multiprocessing to parallelize data collection from the environment, potentially speeding up the process.
-   `train_ppo.py`: Trains the PPO agent by interacting with the real environment.
-   `train_ppo_in_dream.py`: Trains the PPO agent by generating simulated experiences (dreams) from the trained World Model.
-   `play_game.py`: Allows visualizing a trained PPO agent playing in the CarRacing-v3 environment.
-   `create_video.py`: Generates videos of the agent playing or dreaming.
-   `conv_vae.py`: Defines the ConvVAE model architecture.
-   `world_model.py`: Defines the World Model architecture (e.g., WorldModelGRU).
-   `actor_critic.py`: Defines the Actor and Critic network architectures for PPO.
-   `utils.py`, `utils_rl.py`: Contain various helper functions, configurations, and utilities used across the project.

## Setup and Dependencies

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_directory>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The primary dependencies are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
