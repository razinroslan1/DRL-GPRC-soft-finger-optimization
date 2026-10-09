import math
import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box

from fem import FEM_sim

class AngleEnv(Env):
    def __init__(self, mapdl, alpha=1.5, beta=10, epsilon=1.5, total_sims=3072, steps_per_episode=32, gp=None, scaler=None, use_gprc=False):
        super().__init__()

        self.step_counter = 0
        self.total_step_counter = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.mapdl = mapdl
        self.gp = gp
        self.scaler = scaler
        self.use_gprc = use_gprc

        if self.use_gprc and (self.gp is None or self.scaler is None):
            raise ValueError("gp and scaler inputs are necessary when use_gprc=True")

        self.total_sims = total_sims
        self.steps_per_episode = steps_per_episode
        self.epochs = int(math.ceil(self.total_sims / self.steps_per_episode))
        self.current_epoch = 1

        # Upper and lower limits of the three parameters and angle
        self.chamber_height_LL = 8
        self.chamber_height_UL = 18
        self.slit_thickness_LL = 0.8
        self.slit_thickness_UL = 3
        self.slit_length_LL = 1
        self.slit_length_UL= 2

        self.theta_LL = 0
        self.theta_UL = 360

        # Action and observation spaces
        self.low_action = np.array([-1, -1, -1]).astype(np.float32)
        self.high_action = np.array([1, 1, 1]).astype(np.float32)

        self.action_space = Box(low=self.low_action, high=self.high_action, dtype=np.float32)

        self.low_state = np.array([self.theta_LL, self.chamber_height_LL, self.slit_thickness_LL, self.slit_length_LL], dtype=np.float32)
        self.high_state = np.array([self.theta_UL, self.chamber_height_UL, self.slit_thickness_UL, self.slit_length_UL], dtype=np.float32)

        self.observation_space = Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        # Initializing baseline design
        self.chamber_height = 13
        self.slit_thickness = 2
        self.slit_length = 1.5
        
        self.initial_chamber_height = self.chamber_height
        self.initial_slit_thickness = self.slit_thickness
        self.initial_slit_length = self.slit_length

        theta = FEM_sim(self.mapdl, self.chamber_height, self.slit_thickness, self.slit_length, self.gp, self.scaler, self.use_gprc)

        f_ch = (1 - (math.exp((self.chamber_height - self.chamber_height_UL - self.epsilon))))
        f_st = (1 - 2 * (math.exp(-f_ch * self.beta  * (self.slit_thickness - self.slit_thickness_LL))))
        self.initial_objective = theta + self.alpha  * theta * f_st 

        self.state = (theta, self.chamber_height, self.slit_thickness, self.slit_length)

        self.best_reward = self.initial_objective
        self.total_reward = 0
        self.new_best_design = False
        self.best_param_dict = {}
        
        self.initial_reward_check = False

    def step(self, action):
        # Apply action
        self.step_counter += 1
        self.total_step_counter += 1
        state = self.state
        theta, self.chamber_height, self.slit_thickness, self.slit_length = state

        ch_step_size = (max(abs(self.chamber_height_LL - self.initial_chamber_height), abs(self.chamber_height_UL - self.initial_chamber_height))) / (self.steps_per_episode / 2)
        st_step_size = (max(abs(self.slit_thickness_LL - self.initial_slit_thickness), abs(self.slit_thickness_UL - self.initial_slit_thickness))) / (self.steps_per_episode / 2)
        sl_step_size = (max(abs(self.slit_length_LL - self.initial_slit_length), abs(self.slit_length_UL - self.initial_slit_length))) / (self.steps_per_episode / 2)

        self.chamber_height = self.chamber_height + action[0] * ch_step_size
        self.slit_thickness = self.slit_thickness + action[1] * st_step_size
        self.slit_length = self.slit_length + action[2] * sl_step_size
        
        self.chamber_height = np.clip(np.round(self.chamber_height, 3), self.chamber_height_LL, self.chamber_height_UL)
        self.slit_thickness = np.clip(np.round(self.slit_thickness, 3), self.slit_thickness_LL, self.slit_thickness_UL)
        self.slit_length = np.clip(np.round(self.slit_length, 3), self.slit_length_LL, self.slit_length_UL)

        theta = FEM_sim(self.mapdl, self.chamber_height, self.slit_thickness, self.slit_length, self.gp, self.scaler, self.use_gprc)

        info = {}

        self.state = (theta, self.chamber_height, self.slit_thickness, self.slit_length)
        
        terminated = self.slit_thickness <= self.slit_thickness_LL or self.slit_thickness >= self.slit_thickness_UL or self.chamber_height <= self.chamber_height_LL or self.chamber_height >= self.chamber_height_UL

        # Reward function        
        f_ch = (1 - (math.exp((self.chamber_height - self.chamber_height_UL - self.epsilon))))
        f_st = (1 - 2 * (math.exp(-f_ch * self.beta * (self.slit_thickness - self.slit_thickness_LL))))
        
        next_objective = theta + self.alpha * theta * f_st # Summation reward  
        
        previous_objective = self.current_objective
        reward = next_objective - previous_objective

        # Action penalty
        progress = np.clip(((next_objective - self.current_objective) / (self.best_reward - self.current_objective + 1e-6)), 0, 1)

        if self.use_gprc == True:
            penalty_scale = 2
        else:
            penalty_scale = 1

        penalty = penalty_scale * (1 + 2 * progress) * ((action[0]) ** 2 + (action[1]) ** 2 + (action[2]) ** 2)
        
        reward -= penalty
        
        if terminated:
            if self.use_gprc == True:
                reward -= 240
            else:
                reward -= 120

        # Bounding rewards
        if self.use_gprc == True:
            c = 60
        else:
            c = 30

        reward = c * math.tanh(reward / c)
        self.total_reward += reward
            
        param_dict = {'Equivalent sim no.': self.total_step_counter,
                      'Epoch': self.current_epoch,
                      'Step': self.step_counter,
                      'Chamber height': self.state[1],
                      'Slit thickness': self.state[2],
                      'Slit length': self.state[3],
                      'Theta': self.state[0],
                      'Reward': reward,
                      'Penalty': penalty,
                      'Objective': next_objective}
        
        # Saving new best objective
        if next_objective > self.best_reward:
            self.best_reward = next_objective
            self.new_best_design = True
            self.best_param_dict = param_dict
                
        truncated = self.step_counter >= self.steps_per_episode
                
        print(f"Epoch progress: {(self.current_epoch):3d}/{self.epochs:3d} ({(self.total_step_counter / self.total_sims * 100):6.2f}%)")
        print(f"Batch progress: {self.step_counter:3d}/{self.steps_per_episode:3d} ({(self.step_counter / self.steps_per_episode * 100):6.2f}%)")

        print(f"Current run | Chamber height: {float(self.state[1]):6.3f} | Slit thickness: {float(self.state[2]):5.3f} | Slit length: {float(self.state[3]):5.3f} | Theta: {float(self.state[0]):6.3f} | Objective: {float(next_objective):.4f} | Penalty: {float(penalty):.4f} | Reward: {float(reward):.4f} | Eq. sim no.: {self.total_step_counter:4d}")

        if self.new_best_design == True:
            print(f"New best design")
            
        print("")
        self.new_best_design =  False
        
        if truncated or terminated:
            print(f"\n---------------------------------------------------------------------- Total reward: {self.total_reward} | Episode ended at step: {self.step_counter} ----------------------------------------------------------------------")
            print("")
            if self.best_param_dict:
                print(f"Overall best | Chamber height: {self.best_param_dict['Chamber height']:6.3f} | Slit thickness: {self.best_param_dict['Slit thickness']:5.3f} | Slit length: {self.best_param_dict['Slit length']:5.3f} | Theta: {self.best_param_dict['Theta']:6.3f} | Objective: {self.best_param_dict['Objective']:.4f} | Penalty: {self.best_param_dict['Penalty']:.4f} | Reward: {self.best_param_dict['Reward']:.4f} | Eq. sim no.: {self.best_param_dict['Equivalent sim no.']:4d}")
            print("")
            
            self.current_epoch += 1

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        self.chamber_height = 13
        self.slit_thickness = 2
        self.slit_length = 1.5
        if self.initial_reward_check == False:
            self.theta = FEM_sim(self.mapdl, self.chamber_height, self.slit_thickness, self.slit_length, self.gp, self.scaler, self.use_gprc)
        self.state = (self.theta, self.chamber_height, self.slit_thickness, self.slit_length)

        self.initial_reward_check = True
        self.total_reward = 0
        info = {}
        self.current_objective = self.initial_objective
        
        if self.current_epoch != (self.epochs + 1):
            print(f"\nEnvironment reset | Chamber height: {self.chamber_height:.3f} | Slit thickness: {self.slit_thickness:.3f} | Slit length: {self.slit_length:.3f} | Theta: {self.theta:.3f} | Objective: {self.current_objective:.3f}")
        else:
            print("Training complete")
            if self.best_param_dict:
                print(f"Overall top | Chamber height: {self.best_param_dict['Chamber height']:6.3f} | Slit thickness: {self.best_param_dict['Slit thickness']:5.3f} | Slit length: {self.best_param_dict['Slit length']:5.3f} | Theta: {self.best_param_dict['Theta']:6.3f} | Objective: {self.best_param_dict['Objective']:.4f} | Penalty: {self.best_param_dict['Penalty']:.4f} | Reward: {self.best_param_dict['Reward']:.4f} | Eq. sim no.: {self.best_param_dict['Equivalent sim no.']:4d}")

        return np.array(self.state, dtype=np.float32), info