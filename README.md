# Maze---Reinforcement-Learning
First Reinforcement Learning Project

The aim of this project is to teach an agent to learn escape a dynamic maze, where walls, starting position and exit are randomized every episode.
The maze is built using MazeLib library for python, documentation here: https://github.com/john-science/mazelib/blob/main/docs/API.md

This is also my first python project so probably there's a lot of thing that could be improved, or also maybe incorrects.

The teaching steps i thought were to let the agent start learning a complete static maze where maze, start and goal are fixed. Then loading that model and teach the agent to learn a semi-static maze where only the maze is fixed and finally to randomized the full environment (full dynamic maze not implemented yet).

- Environments folder contain the two starting environments (full and semi static)
- Trainings folder contain the firs script where i train the model for the first time (FirstTraining) and the script to load the model and training again on the new environment

--still work in progress--
