# Reinforcement-Learning

This repository contains code implementing examples from:

* Sutton, R. and Barto, A. G. *Reinforcement Learning: An Introduction*. MIT Press, Cambridge, MA, 2015.

To use this code:

### Download

Clone the repository

    git clone https://github.com/louissharrock/Reinforcement-Learning


### Install 

Install a virtual environment

    cd Reinforcement-Learning
    conda env create -f environment.yml
    conda activate venv
    pip install -e .
    

### Basic Functions
    
    import gym
    import gym_gridworld
    env = gym.make('GridWorld-v0')

    
