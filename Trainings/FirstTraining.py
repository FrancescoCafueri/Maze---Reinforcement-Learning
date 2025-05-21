
from stable_baselines3 import PPO
import os
from FullStaticEnv import MazeEnv as StaticEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

n_envs = 1

def make_env():
    env = StaticEnv(rows=3, columns=3, render_mode=False)
    return Monitor(env)

env_fns = [make_env for _ in range(n_envs)]
vec_env = DummyVecEnv(env_fns)

models_dir = "models/PPO_3x3"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=log_dir, ent_coef=0.01, learning_rate= 1e-4)


TIMESTEPS = 10000
for i in range(1, 15):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_3x3")
    model.save(f"{models_dir}/{TIMESTEPS * i}")

