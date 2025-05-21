import gymnasium as gym
from stable_baselines3 import PPO
import os
from SemiStaticEnv import MazeEnv as SemiStaticEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

models_dir_to_save = "models/PPO_3x3"


logdir = "logs"

models_path = f"models/PPO_3x3/140000.zip"



if not os.path.exists(models_dir_to_save):
    os.makedirs(models_dir_to_save)
if not os.path.exists(logdir):
    os.makedirs(logdir)
n_envs = 20
def make_env():
    env = SemiStaticEnv(rows=3, columns=3, render_mode=False)
    return Monitor(env)

env_fns = [make_env for _ in range(n_envs)]
vec_env = DummyVecEnv(env_fns)



model = PPO.load(models_path, env=vec_env, verbose=1, tensorboard_log=logdir, ent_coef=1e-2, lr = 1e-4)

TIMESTEPS = 10000

for i in range(1, 15):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_3x3")
    model.save(f"{models_dir_to_save}/{TIMESTEPS*i}")
