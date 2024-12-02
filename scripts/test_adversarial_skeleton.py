import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import rfrl_gym
import time
import datagen.liquid.spectrum
import sys
sys.path.insert(0, './src/detection')

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='wifi_scenario.json', type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-m', '--gym_mode', default='adversarial', type=str, help='Which type of RFRL gym environment to run.')
parser.add_argument('-e', '--epochs', default=5, type=int, help='Number of training epochs. (def: %(default)s)')
args = parser.parse_args()

if args.gym_mode == 'abstract':
    env = gym.make('rfrl-gym-abstract-v0', scenario_filename=args.scenario)
elif args.gym_mode == 'iq':
    env = gym.make('rfrl-gym-iq-v0', scenario_filename=args.scenario)
elif args.gym_mode == 'adversarial':
    env = gym.make('rfrl-gym-adversarial-v0', scenario_filename=args.scenario)

obs, info = env.reset()
terminated = truncated = False
for i in range(args.epochs):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
