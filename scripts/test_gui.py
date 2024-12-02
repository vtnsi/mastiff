import sys
sys.path.insert(0, './src/detection')

import argparse
import gymnasium as gym
import rfrl_gym

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='zigbee_scenario.json', type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-e', '--epochs', default=5, type=int, help='Number of training epochs. (def: %(default)s)')
args = parser.parse_args()

env = gym.make('rfrl-gym-adversarial-v0', scenario_filename=args.scenario)
obs, info = env.reset()

terminated = truncated = False
for i in range(args.epochs):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()