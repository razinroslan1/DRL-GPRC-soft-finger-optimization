from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

from environment import AngleEnv
from gprc import configure_gprc

from ansys.mapdl.core import launch_mapdl

def main():
    mapdl = launch_mapdl(nproc = 6, # Number of CPU cores utilized
                         mode = "grpc")
    print(f"\nMAPDL Solver launched")
    
    use_gprc = True # Enable or disable GPRC
    seed = 42       # Random seed

    gp, scaler = configure_gprc(use_gprc, seed)

    env = AngleEnv(mapdl,                       # MAPDL solver
                   alpha = 1.5,                 # Slit thickness penalization weighting
                   beta = 10,                   # Slit thickness penalization agressiveness
                   epsilon = 1.5,               # Chamber height penalization agressiveness
                   total_sims = 3072,           # Total number of simulations
                   steps_per_episode = 32,      # Number of steps per episode
                   gp = gp,                     # Gaussian Process model
                   scaler = scaler,             # Scaler for normalizing GPRC input parameters
                   use_gprc = use_gprc)         # Enable or disable GPRC

    callback = StopTrainingOnMaxEpisodes(max_episodes = env.epochs, # Maximum number of episodes
                                         verbose = 0) 

    model = PPO(policy = "MlpPolicy",       # MLP
                env = env,                  # DRL environment
                learning_rate = 2.5e-4,     # Optimizer learning rate
                n_steps = 128,              # Number of steps collected for each rollout
                batch_size = 64,            # Minibatch size
                n_epochs = 10,              # Number of optimization epochs for each rollout
                gamma = 0.99,               # Discount factor
                gae_lambda = 0.95,          # Bias vs variance trade-off factor
                clip_range = 0.2,           # Clipping parameter
                ent_coef = 0.0,             # Entropy coefficient
                vf_coef = 0.5,              # Value-function coeffcieint
                max_grad_norm = 0.5,        # Maximum gradient clipping value
                verbose = 0,                #  
                device = 'cuda',            #
                seed = seed)                # Random seed

    model.learn(total_timesteps = 1000000, 
                callback = callback)

if __name__ == "__main__":
    main()