import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import rfrl_gym
import time
import datagen.liquid.spectrum
import sys
sys.path.insert(0, './src/detection')
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='zigbee_scenario.json', type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-m', '--gym_mode', default='adversarial', type=str, help='Which type of RFRL gym environment to run.')
parser.add_argument('-a', '--agent_model', default='mastiff_gym_ppo_intermediate', type=str, help='Which RL agent to test')
parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of training epochs.')
args = parser.parse_args()

if __name__ == "__main__":
    if args.gym_mode == 'adversarial':
        env = gym.make('rfrl-gym-adversarial-v0',scenario_filename=args.scenario)

    model = PPO("MlpPolicy", env, verbose=1, n_steps=env.max_steps)
    model = model.learn(total_timesteps= env.max_steps * args.epochs,log_interval=10, progress_bar=True)
    env.reset()
    model.save(args.agent_model)

    del model

    model = PPO.load(args.agent_model)
    obs, info = env.reset()
    terminated = truncated = False
    while not terminated and not truncated:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()
