import gymnasium as gym
from gymnasium import spaces

from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms
import numpy as np
import random


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class MazeEnv(gym.Env):
    def __init__(self, rows, columns, render_mode=False):

        self.rows, self.columns = rows, columns
        

        self.grid_actual = self.generate_maze()

        # defining max rows and columns
        self.max_rows, self.max_columns = 16, 16

        # setting real grid size since mazelib generates a bigger maze (Ex. input 3x3 --> maze: 5x5)
        self.n_rows = len(self.grid_actual)
        self.n_columns = len(self.grid_actual[0]) if self.n_rows > 0 else 0

        self.render_mode = render_mode
        if self.render_mode:
            self.fig, self.ax = plt.subplots()

        # looking for free cells and setting start and goal positions
        free_cells = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_columns)
            if self.grid_actual[r][c] == 0
        ]
        self.start_pos, self.goal_pos = random.sample(free_cells, 2)
        self.agent_pos = self.start_pos

        # normalizing agent and target positions for the padding grid
        self.agent_norm  = np.array(self.start_pos, dtype=np.float32)  / [self.max_rows-1, self.max_columns-1]
        self.target_norm = np.array(self.goal_pos,  dtype=np.float32)  / [self.max_rows-1, self.max_columns-1]

        # max step for timeout
        self.max_step = 50

        # visisted cells set
        self.visited = set()

        #action space
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),
        }
        
        # obs space
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, shape=(self.max_rows, self.max_columns), dtype=np.float32),
                "agent": spaces.Box(0.0, 1.0, shape=(2,),      dtype=np.float32),
                "target":spaces.Box(0.0, 1.0, shape=(2,),      dtype=np.float32),
            }
        )

    def step(self, action):

        # timeout
        self.step_count += 1
        if self.step_count >= self.max_step:
            
            return self._get_obs(), 0.0, False, True, {}
        
        move = self._action_to_direction[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        terminated = False
        reward = 0

        if (0 <= new_pos[0] < self.n_rows and
            0 <= new_pos[1] < self.n_columns and
            self.grid_actual[new_pos[0], new_pos[1]] == 0):
            
            # valid action
            self.agent_pos = new_pos

            if np.array_equal(self.agent_pos, self.goal_pos):
                reward += 1
                terminated = True
        
        info = self._get_info()
        obs = self._get_obs()
        #self.render()
        return obs, reward, terminated, False, info
            
        '''self.step_count += 1
        if self.step_count >= self.max_step:
            
            return self._get_obs(), 0.0, False, True, {}
        
        # distance before action
        prev_dist = np.linalg.norm(
            np.array(self.agent_pos, dtype=np.float32) -
            np.array(self.goal_pos,  dtype=np.float32)
        )

        # checking action validity
        move = self._action_to_direction[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        if (0 <= new_pos[0] < self.n_rows and
            0 <= new_pos[1] < self.n_columns and
            self.grid_actual[new_pos[0], new_pos[1]] == 0):
            
            # valid action
            self.agent_pos = new_pos

            # goal reached
            if np.array_equal(self.agent_pos, self.goal_pos):
                reward = +10.0
                terminated = True
            # new cell visited
            elif self.agent_pos not in self.visited:
                self.visited.add(self.agent_pos)
                reward = +0.05
                terminated = False
            # already visited cell
            else:
                reward = 0
                terminated = False
        # invalid action
        else:  
            reward = -0.25
            terminated = False

        # penalize for each step
        reward -= 0.005

        # shaping reward based on distance
        new_dist = np.linalg.norm(
            np.array(self.agent_pos, dtype=np.float32) -
            np.array(self.goal_pos,  dtype=np.float32)
        )
        shaping_coef   = 0.1
        shaping_reward = shaping_coef * (prev_dist - new_dist)
        reward        += shaping_reward

        
        info = self._get_info()
        info["shaping_reward"] = shaping_reward
        #self.render()


        return self._get_obs(), reward, terminated, False, info'''
        


        
    def reset(self, seed = None):
        super().reset(seed=seed)
        
        self.grid_full = np.zeros((self.max_rows, self.max_columns), dtype=np.float32)
        self.grid_full[:self.n_rows, :self.n_columns] = self.grid_actual

        self.agent_pos = self.start_pos

        self.visited = {self.agent_pos}

        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        self.prev_distance = info["distance"]

        return obs, info
    
    def render(self, mode='human'):
        if not self.render_mode:
            return
        display_grid = self.grid_actual.copy()
        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        display_grid[ar, ac] = 2
        display_grid[gr, gc] = 3

        cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'green'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        self.ax.clear()
        self.ax.imshow(display_grid, cmap=cmap, norm=norm)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.ax.set_title("Maze Render")
        self.fig.canvas.draw()
        plt.pause(0.01)

        
    # maze generation
    def generate_maze(self):
        
        m = Maze()
        m.generator = DungeonRooms(self.rows, self.columns)
        m.generate()

        return m.grid
        
    
    
    def _get_obs(self):

        return {
        "grid": self.grid_full,
        "agent": self.agent_norm,
        "target": self.target_norm}
        
    
    def _get_info(self):
        distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
        return {"distance": distance}
    

gym.register(
    id="Maze-static",
    entry_point="FullStaticEnv:StaticEnv",
)


                
    
        