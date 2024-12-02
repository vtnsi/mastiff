import gymnasium as gym
import time
import json
import numpy as np
from PyQt6.QtWidgets import QApplication
import rfrl_gym.renderers
import rfrl_gym.entities
import rfrl_gym.signals
import rfrl_gym.spectrums
import torch
from detection.detector import *
from collections import defaultdict

class RFRLGymAdversarialEnv(gym.Env):
    metadata = {'render_modes': ['null', 'terminal', 'pyqt'], 'render_fps':100,
                    'action_modes': ['power', 'power_and_bw']}

    def __init__(self, scenario_filename, num_episodes=1, num_bursts=1):   
        self.num_episodes = num_episodes
        self.num_bursts = num_bursts

        # Load in the JSON scenario file and check for valid entries.
        f_idx = open('scenarios/' + scenario_filename)
        self.scenario_metadata = json.load(f_idx)
        self.__validate_scenario_metadata()
        
        # Get the environment parameters from the scenario file.
        self.num_channels = self.scenario_metadata['environment']['num_channels']
        self.max_steps = self.scenario_metadata['environment']['max_steps']
        self.action_mode = self.scenario_metadata['environment']['action_mode']

        # Get the entity parameters from the scenario file and initialize the entities based on which spectrum they belong to
        spectrum_idx = 0
        self.num_entities = 0
        self.spectrum_list = []
        for spectrum in self.scenario_metadata['spectrums']:
            spectrum_idx += 1
            obj_str = 'rfrl_gym.spectrums.spectrum.' + 'RF_' + self.scenario_metadata['spectrums'][spectrum]['type'] + '(spectrum_label=\'' + str(spectrum) + '\', num_channels=' + str(self.num_channels) + ', '
            for param in self.scenario_metadata['spectrums'][spectrum]:
                if not param == 'type':
                    obj_str += (param + '=' + str(self.scenario_metadata['spectrums'][spectrum][param]) + ', ')
            obj_str += ('scenario_metadata=' + str(self.scenario_metadata) + ', ')
            obj_str += ')'
            self.spectrum_list.append(eval(obj_str))
            self.spectrum_list[-1].set_spectrum_index(spectrum_idx)
            self.num_entities += self.spectrum_list[-1].signal_idx
        
        # Get the render parameters from the scenario file and initialize the render if necessary.
        self.render_mode = self.scenario_metadata['render']['render_mode']
        self.render_fps = self.scenario_metadata['render']['render_fps']
        self.next_frame_time = 0

        if self.render_mode == 'pyqt':
            self.pyqt_app = QApplication([])

        # Set the gym's valid action spaces.
        self.power_options = np.linspace(-20,20,11)
        self.power_range = [(x-np.min(self.power_options))/(np.max(self.power_options)-np.min(self.power_options)) for x in self.power_options]
        self.bw_options = [100e3,500e3,2e6,5e6,10e6,20e6]
        self.target_bw = 2e6 
        if self.action_mode == 'power':
            self.action_space = gym.spaces.Discrete(len(self.power_options))     
        elif self.action_mode == 'power_and_bw':
            self.action_space = gym.spaces.MultiDiscrete([len(self.power_options),len(self.bw_options)])

        # Set the gym's valid observation space.
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(233,256), dtype=np.float32)

    def sensor(self, matrix):
        # return processed observation
        tens = torch.from_numpy(matrix).to(self.device).type(torch.half)
        observation = self.detect.detect_single(tens)
        return observation

    def evaluate_sensor(self,observation,bounds):
        # detector can use observation (i.e. ML sensor outputs) + bounds (i.e. datagen ground truth) to get bbox_iou
        # observation is already tensor
        # need to convert bounds to tensor
        # need to artifically inflate bounds to match gt training?
        bounds = torch.from_numpy(np.array(bounds)).to(self.device).type(torch.half)
        # gns is the image size by which we need to normalize, 640 here
        gns = 640

        obs_iou_final_list = []
        obs_matching_gt = []
        
        # check if any detections first
        if len(observation):
            for *xyxy1, conf, cls in reversed(observation):
                iou_temp_list = []
                # [x1, y1, x2, y2] xy1=top-left, xy2=bottom-right for observation, i.e. ML sensor output
                obs_tens = torch.tensor(xyxy1).view(1, 4)/gns
                bounds_tens_num = 0
                for cls, *xxyy2 in bounds:
                    bounds_tens_num+=1
                    bounds_tens = self.xxyy2xyxy_inflate(torch.tensor(xxyy2).view(1, 4)).view(-1)
                    # bbox_iou takes tensors
                    # trying 1 box comparison first, will iterate over all ultimately
                    iou = self.detect.bbox_iou(obs_tens, bounds_tens)
                    iou = iou.view(-1)
                    iou = iou.numpy()
                    iou_temp_list.append(iou[0])

                obs_iou_final_list.append(max(iou_temp_list))
                obs_matching_gt.append(np.argmax(iou_temp_list))
                #matching_gt.append(argmax(iou_tem_list)) # gotta fix this to give index of ground truth signal matching the largest iou for that observation box
        obs_sensor_metric = obs_iou_final_list
        # return metric for reward
        return obs_sensor_metric, obs_matching_gt

    def xxyy2xyxy_inflate(self, x):
        # Convert nx4 boxes from [x1, x2, y1, y2] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0]-0.02  # x1
        y[:, 1] = x[:, 2]-0.01 # y1
        y[:, 2] = x[:, 1]+0.02 # x2
        y[:, 3] = x[:, 3]+0.01 # y2
        return y
    
    def calculate_reward(self, sensor_metric):
        # return reward
        tpr = 0.0
        fpr = 0.0
        if self.action_mode == 'power_and_bw':
            IoU_thresh = 0.1
        elif self.action_mode == 'power':
            IoU_thresh = 0.61
        IoU = sensor_metric[0]
        gt = sensor_metric[1]
        tp = []
        fp = []

        output = defaultdict(list)
        list_of_tuples = list(zip(gt,IoU))
        for k,v in list_of_tuples:
            output[k].append(v)

        if len(output) != 0:
            for i in range(len(output)):
                if i in output.keys():
                    max_detect = np.max(output[i])
                    if max_detect >= IoU_thresh:
                        tp.append(max_detect)
                    for j in range(len(output[i])):
                        if output[i][j] not in tp:
                            if output[i][j] != 0.0:
                                fp.append(output[i][j])
        tpr += np.sum(len(tp))
        fpr += np.sum(len(fp))
        if self.action_mode == 'power_and_bw':
            if self.bw_action == self.target_bw:
                bw = 1
            else:
                bw = 0
            r = -0.5*(np.sum(tp)) - 0.25*bw - 0.25*self.power_range[self.action_idx]
        elif self.action_mode == 'power':
            r = -0.5*(np.sum(tp)) - 0.5*self.power_range[self.action_idx]
        return r, tpr, fpr
    
    def get_spectrum_observation(self, spectrum):
        if self.action_mode == 'power':
            spectrum._get_spectrum(self.power_action, '')
        elif self.action_mode == 'power_and_bw':
            spectrum._get_spectrum(self.power_action, self.bw_action)
        fig,matrix,timesteps,freqs,samples,bounds = spectrum._step()
        return matrix,bounds,timesteps
    
    def process_matrix(self, matrix):
        input_tensor = torch.from_numpy(matrix).to(self.device).type(torch.half)
        np.shape(input_tensor)
        img_tensor = self.detect.preprocess_single(input_tensor)
        img_tensor.half()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    
    def get_observation_and_reward(self,spectrum):
        reward = 0.0
        tpr = 0.0
        fpr = 0.0
        spectrum._reset()
        try:
            for burst in range(self.num_bursts):
                result = self.get_spectrum_observation(spectrum)

                # Sensor goes here (feed raw IQ into sensor)
                matrix = result[0]
                bounds = result[1]
                observation = self.sensor(matrix)

                temp_matrix = self.process_matrix(matrix)
                if self.info['step_number'] == 1:
                    self.info['spectrum'] = matrix
                    self.info['bounding_boxes'] = bounds
                    self.info['observation'] = observation
                else:            
                    self.info['spectrum'] = np.concatenate((self.info['spectrum'],matrix))
                    self.info['bounding_boxes'] = bounds
                    self.info['observation'] = observation
            
                print(matrix.shape)

                # Calculate the player reward based on sensor results
                sensor_metric = self.evaluate_sensor(observation,bounds)
                reward_results = self.calculate_reward(sensor_metric)
                reward += reward_results[0]
                tpr += reward_results[1]
                fpr += reward_results[2]
        finally:
            spectrum._close()
        
        return matrix, reward, tpr, fpr

    def step(self, action):
        if self.action_mode == 'power':
            self.action_idx = action
            self.power_action = self.power_options[action]
        elif self.action_mode == 'power_and_bw':
            self.action_idx = action[0]
            self.power_action = self.power_options[action[0]]
            self.bw_action = self.bw_options[action[1]]

        self.info['step_number'] += 1
        if self.action_mode == 'power':
            self.info['action_history'][0][self.info['step_number']] = action
            self.info['power_history'][self.info['step_number']] = self.power_action
        elif self.action_mode == 'power_and_bw':
            self.info['action_history'][0][self.info['step_number']] = action[0]
            self.info['action_history'][1][self.info['step_number']] = action[1]
            self.info['power_history'][self.info['step_number']] = self.power_action
            self.info['bw_history'][self.info['step_number']] = self.bw_action

        # Get spectrum observation (raw IQ) from datagen
        for spectrum in self.spectrum_list:
            ans = self.get_observation_and_reward(spectrum)
            observation = ans[0]
            reward = ans[1]
            tpr = ans[2]
            fpr = ans[3]
        
        self.info['reward_history'][self.info['step_number']] = reward
        self.info['cumulative_reward'][self.info['step_number']] = np.sum(self.info['reward_history'])
        self.info['tpr_history'][self.info['step_number']] = tpr
        self.info['fpr_history'][self.info['step_number']] = fpr
        self.info['cumulative_tpr'][self.info['step_number']] = np.sum(self.info['tpr_history'])
        self.info['cumulative_fpr'][self.info['step_number']] = np.sum(self.info['fpr_history'])

        # Update return variables and run the render.
        done = False
        if self.info['step_number'] == self.max_steps:
            self.info['episode_reward'] = np.append(self.info['episode_reward'], self.info['cumulative_reward'][self.info['step_number']])
            done = True

        return observation, reward, done, done, self.info

    def reset(self, options={'reset_type':'soft'}, seed=None):
        # Temporarily store episode specific variables if they exist.
        if hasattr(self, 'info') and options['reset_type'] == 'soft':
            episode_number = self.info['episode_number']
            episode_reward = self.info['episode_reward']        
        else:
            episode_number = -1
            episode_reward = np.array([], dtype=float)
            if self.render_mode == 'pyqt':
                self.renderer = rfrl_gym.renderers.pyqt_renderer.PyQtRenderer()
            if self.render_mode != 'null':
                self.renderer.reset()
     
        # Reset the gym info dictionary and if necessary restore episode variables.
        self.info = {}
        self.info['step_number'] = 0
        self.info['num_entities'] = self.num_entities
        self.info['num_episodes'] = self.num_episodes
        self.info['episode_reward'] = episode_reward  
        self.info['episode_number'] = episode_number + 1   
        self.info['action_history'] = np.zeros((2, self.max_steps+1), dtype=int)
        self.info['true_history'] = np.zeros((self.max_steps+1, self.num_channels), dtype=int)
        self.info['observation_history'] = np.zeros((self.max_steps+1, self.num_channels), dtype=int)
        self.info['reward_history'] = np.zeros(self.max_steps+1, dtype=float)
        self.info['cumulative_reward'] = np.zeros(self.max_steps+1, dtype=float)
        self.info['tpr_history'] = np.zeros(self.max_steps+1, dtype=float)
        self.info['fpr_history'] = np.zeros(self.max_steps+1, dtype=float)
        self.info['cumulative_tpr'] = np.zeros(self.max_steps+1, dtype=float)
        self.info['cumulative_fpr'] = np.zeros(self.max_steps+1, dtype=float)
        self.info['power_history'] = np.zeros(self.max_steps+1, dtype=float)
        self.info['bw_history'] = np.zeros(self.max_steps+1, dtype=float)
        self.info['spectrum'] = np.empty((233,256))
        self.info['bounding_boxes'] = []
        self.info['observation'] = []
        self.info['max_steps'] = self.max_steps

        for spectrum in self.spectrum_list:
            spectrum._reset()
            self.info['spectrum_duration'] = spectrum.observation_duration
            self.info['spectrum_observation'] = spectrum.observation_bandwidth

        # initialize MUT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.action_mode == 'power':
            self.weights = 'src/detection/best.pt'
        elif self.action_mode == 'power_and_bw':
            self.weights = 'src/detection/best_zigbee.pt'
        self.detect = detector(weights=self.weights,device=self.device)
        self.detect.setup()
        
        if self.render_mode == 'pyqt':
            self.renderer.reset()

        # Set return variables.  
        observation = self.observation_space.sample()
        return observation, {}
        
    def render(self):
        if self.render_mode != 'null':
            if self.info['step_number'] == 0:
                self.next_frame_time = time.time()

            if ((time.time()-self.next_frame_time) < (1.0/self.render_fps)):
                time.sleep(time.time()-self.next_frame_time)                        
            self.next_frame_time = time.time()
            self.renderer.render(self.info)
        return

    def close(self):   
        input('Press Enter to end the simulation...')
        return

    def __validate_scenario_metadata(self):
        # Validate scenario environment parameters.
        assert self.scenario_metadata['environment']['num_channels'] > 0, 'Environment parameter \'num_channels\' is invalid.'
        assert self.scenario_metadata['environment']['max_steps'] > 0, 'Environment parameter \'max_steps\' is invalid.'
        assert self.scenario_metadata['environment']['action_mode'] in self.metadata['action_modes'], 'Invalid action mode. Must be one of the following options: {}'.format(self.metadata["action_modes"])
        
        # Validate scenario render parameters.
        assert self.scenario_metadata['render']['render_mode'] is None or self.scenario_metadata['render']['render_mode'] in self.metadata['render_modes'], 'Invalid render mode. Must be one of the following options: {}'.format(self.metadata["render_modes"])
        assert self.scenario_metadata['render']['render_fps'] > 0, 'Render parameter \'render_fps\' is invalid.'
        assert self.scenario_metadata['render']['render_history'] > 0, 'Render parameter \'render_history\' is invalid.'
