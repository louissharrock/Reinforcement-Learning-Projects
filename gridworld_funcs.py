# import packages
import gym
import numpy as np
import random
import copy

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# nice arrows
arrow_dict = {"left": '\u2190', "down": '\u2193', "right": '\u2192', "up": '\u2191'}


# functions #


# (randomly chosen) deterministic policy
def get_policy(env):
    policy = []
    for s in range(env.n_states):
        p_s = [0] * env.n_actions
        random_action = random.choice(env.get_actions())
        p_s[env.action_to_ind(random_action)] = 1
        policy.append(p_s)
    return policy


# random (uniform probabilities) policy
def get_uniform_policy(env):
    policy = []
    for s in range(env.n_states):
        p_s = [0.25] * env.n_actions
        policy.append(p_s)
    return policy

# make greedy policy
def make_greedy_policy(env,policy,epsilon):
    policy_epsilon = []
    for s in range(env.n_states):
        p_s = [epsilon/env.n_actions]*env.n_actions
        p_s_max_index = np.argmax(policy[s])
        p_s[p_s_max_index] += 1 - epsilon
        policy_epsilon.append(p_s)
    return policy_epsilon


# fetch action for a given state & policy
def get_action(env, policy, state):
    state_ind = env.state_to_ind(state)
    action_prob_dist = policy[state_ind]
    action = np.random.choice(env.get_actions(), 1, p=action_prob_dist)
    return action[0]


# print actions
def print_actions(env, policy):
    col_width = 2
    for s in range(env.n_states):
        state = env.ind_to_state(s)
        action_s = get_action(env, policy, state)
        print(arrow_dict[action_s].center(col_width), end=" ")
        if state[1] == env.ncol - 1:
            print()
    print("\n")


# simulate an episode
def simulate_episode(env, init_state, policy, n_moves=5,print_moves=False,exploring_start=False):
    env = gym.make("GridWorld-v0",state=init_state,nrow=env.nrow,ncol=env.ncol,
                   target_state_1 = env.target_state_1,target_state_2=env.target_state_2)
    env.reset()
    if(print_moves):
        print("The initial state:")
        env.render()
    total_reward = 0
    episode = []
    for i in range(n_moves):
        state = env.state
        if i==0:
            if exploring_start:
                action = random.choice(env.get_actions())
            else:
                action = get_action(env,policy,state)
        else:
            action = get_action(env, policy, state)
        new_state, reward, action, prob = env.step(action)

        episode.append([list(state), action, reward])
        total_reward += reward
        if(print_moves):
            print("The next state:")
            env.render()
    return episode, total_reward


# print episode
def print_episode(episode, total_reward):
    for i in range(len(episode)):
        state, action, reward = episode[i][0], episode[i][1], episode[i][2]
        print("S_", i, " = ", state, ", A_", i, " = ", arrow_dict[action], ", R_", i+1, " = ", reward, sep="")


# evaluate value function for a given policy (single state)
def evaluate_single_policy(env, values, policy, state, discount_rate, actions=None, deterministic=True):
    value_s = 0
    if actions is None:
        if deterministic is True:
            actions = get_action(env, policy, state)
        else:
            actions = env.get_actions()

    if deterministic is True:
        for transition in env.probs[env.state_to_ind(state)][actions]:
            p_next_state = transition[0]
            s_next_state = transition[1]
            r_next_state = transition[2]
            value_s += p_next_state * (r_next_state + discount_rate * values[env.state_to_ind(s_next_state)])

    else:
        for action in actions:
            action_prob = policy[env.state_to_ind(state)][env.action_to_ind(action)]
            for transition in env.probs[env.state_to_ind(state)][action]:
                p_next_state = transition[0]
                s_next_state = transition[1]
                r_next_state = transition[2]
                value_s += action_prob * p_next_state * (
                        r_next_state + discount_rate * values[env.state_to_ind(s_next_state)])
    return value_s


# evaluate value function for a given policy (all states)
def evaluate_all_policy(env, values, policy, discount_rate, n_sweeps, deterministic=True):
    if values is None:
        values = [0] * env.n_states
    all_values = np.zeros((n_sweeps,env.n_states))
    counter = 0
    while counter < n_sweeps:
        for s in range(env.n_states):
            state = env.ind_to_state(s)
            values[s] = evaluate_single_policy(env, values, policy, state, discount_rate, deterministic=deterministic)
        all_values[counter, :] = values
        counter += 1

    return values, all_values


# print actions
def print_actions(env, policy):
    col_width = 2
    for s in range(env.n_states):
        state = env.ind_to_state(s)
        action_s = get_action(env, policy, state)
        print(arrow_dict[action_s].center(col_width), end=" ")
        if state[1] == env.ncol - 1:
            print()
    print()


# print value function
def print_values(env, values):
    col_width = 5
    for s in range(env.n_states):
        state = env.ind_to_state(s)
        value_s = str(np.round(values[s], 1))
        print(value_s.center(col_width), end=" ")
        if state[1] == env.ncol - 1:
            print()
    print()

def plot_values(env,values):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x,y = np.meshgrid(range(env.nrow),range(env.ncol))
    z = np.array(values).reshape(env.nrow,env.ncol)
    ax.plot_surface(x,y,z)
    plt.show()

def plot_values_animation(env,all_values,n_episodes):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_zlim(np.min(all_values),np.max(all_values))
    x,y = np.meshgrid(range(env.nrow),range(env.ncol))
    for i in range(n_episodes):
        z = np.array(all_values[i,:]).reshape(env.nrow,env.ncol)
        plot = ax.plot_surface(x,y,z,color="C0")
        plt.pause(0.05)
        plot.remove()
    plt.show()


# improve policy for a given value function (single state)
def improve_single_policy(env, values, policy, state, discount_rate):
    v_s_max = float("-inf")
    for action in env.get_actions():
        v_s = evaluate_single_policy(env, values, policy, state, discount_rate, action, deterministic=True)
        if v_s > v_s_max:
            v_s_max = v_s
            action_max = action
    return action_max


# improve policy for a given value function (all states)
def improve_all_policy(env, values, policy, discount_rate):
    is_policy_stable = True
    new_policy = copy.deepcopy(policy)
    for s in range(env.n_states):
        current_state = env.ind_to_state(s)
        current_action = get_action(env, policy, current_state)
        action_max = improve_single_policy(env, values, policy, current_state, discount_rate)
        # print("State:",current_state,"Action:",current_action,"New action",action_max)
        if action_max != current_action:
            is_policy_stable = False
            new_policy[s] = [0] * env.n_actions
            new_policy[s][env.action_to_ind(action_max)] = 1
    return new_policy, is_policy_stable


# policy evaluation & policy iteration
def dynamic_program(env, values, policy, discount_rate, n_eval):
    current_policy = policy
    current_values = values
    is_policy_stable = False
    counter = 0
    while is_policy_stable is False:
        counter += 1
        current_values, current_values_all = evaluate_all_policy(env, current_values, current_policy, discount_rate, n_eval)
        current_policy, is_policy_stable = improve_all_policy(env, current_values, current_policy, discount_rate)
    return current_policy, current_values, counter


# estimate value function using monte carlo
def evaluate_all_policy_monte_carlo(env,values,policy,n_moves,discount_rate,n_episodes,
                                    init_state=None,deterministic=True):
    value_function = values
    value_function_all_eps = np.zeros((n_episodes,env.n_states))
    returns_sum = [0]*env.n_states
    returns_count = [0]*env.n_states

    for i in range(n_episodes):
        if init_state is None:
            init_state_i_ind = random.randint(0,env.n_states-1)
            init_state_i = env.ind_to_state(init_state_i_ind)
        if deterministic is False:
            policy = get_uniform_policy(env)
        episode, total_reward = simulate_episode(env, init_state_i,policy, n_moves)
        episode_states = [episode[i][0] for i in range(n_moves)]
        for state in episode_states:
            first_occurence = next(count for count,value in enumerate(episode) if value[0]==state)
            G = sum([value[2]*(discount_rate**count) for count,value in enumerate(episode[first_occurence:])])
            returns_sum[env.state_to_ind(state)] += G
            returns_count[env.state_to_ind(state)] += 1
            value_function[env.state_to_ind(state)] = returns_sum[env.state_to_ind(state)]/returns_count[env.state_to_ind(state)]
        value_function_all_eps[i,:] = value_function

    return value_function, value_function_all_eps


# estimate action value function using monte carlo
def evaluate_all_action_policy_monte_carlo(env, action_values, policy, n_moves, discount_rate, n_episodes,
                                           init_state=None, deterministic=True, exploring_start=True,
                                           epsilon=None):
    action_value_function = action_values
    action_value_function_all_eps = np.zeros((n_episodes, env.n_states,env.n_actions))
    returns_sum = np.zeros((env.n_states,env.n_actions))
    returns_count = np.zeros((env.n_states,env.n_actions))

    if epsilon is not None:
        policy = make_greedy_policy(env, policy, epsilon)

    for i in range(n_episodes):
        if init_state is None:
            init_state_i_ind = random.randint(0, env.n_states - 1)
            init_state_i = env.ind_to_state(init_state_i_ind)
        if deterministic is False:
            policy = get_uniform_policy(env)
        episode, total_reward = simulate_episode(env, init_state_i, policy, n_moves,exploring_start=exploring_start)
        episode_state_actions = [(episode[i][0],episode[i][1]) for i in range(n_moves)]
        #print_episode(episode,total_reward)
        for state_action in episode_state_actions:
            state = state_action[0]
            action = state_action[1]
            first_occurence = next(count for count, value in enumerate(episode) if (value[0] == state and value[1] == action))
            #print(state_action)
            #print(first_occurence)
            G = sum([value[2] * (discount_rate ** count) for count, value in enumerate(episode[first_occurence:])])
            #print(G)
            returns_sum[env.state_to_ind(state),env.action_to_ind(action)] += G
            returns_count[env.state_to_ind(state),env.action_to_ind(action)] += 1
            action_value_function[env.state_to_ind(state),env.action_to_ind(action)] = returns_sum[env.state_to_ind(state),env.action_to_ind(action)] / returns_count[
                env.state_to_ind(state),env.action_to_ind(action)]
        #print(action_value_function)
        action_value_function_all_eps[i, :, :] = action_value_function

    return action_value_function, action_value_function_all_eps, returns_sum, returns_count


def improve_all_action_policy_monte_carlo(env,action_values,policy):
    is_policy_stable = True
    new_policy = copy.deepcopy(policy)
    for s in range(env.n_states):
        current_state = env.ind_to_state(s)
        current_action = get_action(env, policy, current_state)
        current_action_ind = env.action_to_ind(current_action)
        max_action_ind = np.argmax(action_values[s, :])
        if max_action_ind != current_action_ind:
            is_policy_stable = False
            new_policy[s] = [0] * env.n_actions
            new_policy[s][max_action_ind] = 1
    return new_policy

def general_policy_iteration_monte_carlo(env, action_values, policy, discount_rate, n_moves,
                                         n_episodes, n_iter, deterministic=True, exploring_start = True,
                                         epsilon = None):
    current_policy = policy
    current_action_values = action_values
    for i in range(n_iter):
        current_action_values, current_action_values_all, returns_sum, returns_count = \
            evaluate_all_action_policy_monte_carlo(env, current_action_values, current_policy,
                                                   n_moves, discount_rate, n_episodes,
                                                   init_state=None, deterministic=deterministic,
                                                   exploring_start = exploring_start,
                                                   epsilon = epsilon)
        current_policy = improve_all_action_policy_monte_carlo(env, current_action_values, current_policy)
        #print_actions(env,current_policy)
    return current_policy, current_action_values


def monte_carlo_ES(env,action_values,policy,discount_rate,n_moves,n_episodes,deterministic=True,
                   exploring_start = True, epsilon = None):
    current_policy = policy
    returns_sum = np.zeros((env.n_states,env.n_actions))
    returns_count = np.zeros((env.n_states,env.n_actions))
    current_action_values_all = np.zeros((n_episodes,env.n_states,env.n_actions))

    for i in range(n_episodes):
        current_action_values, current_action_values_all_tmp, current_returns_sum, current_returns_count = \
            evaluate_all_action_policy_monte_carlo(env, action_values, current_policy,
                                                   n_moves, discount_rate, n_episodes = 1,
                                                   init_state=None, deterministic=deterministic,
                                                   exploring_start = exploring_start,
                                                   epsilon = epsilon)
        # average and accumulate all returns
        returns_sum += returns_sum
        returns_count += returns_count

        for j in range(env.n_states):
            for k in range(env.n_actions):
                if returns_count[j,k]!=0:
                    current_action_values[j,k] = returns_sum[j,k] / returns_count[j,k]

        current_action_values_all[i,:,:] = current_action_values

        current_policy = improve_all_action_policy_monte_carlo(env, current_action_values, current_policy)

    return current_policy, current_action_values

def action_values_to_state_values(env,action_values):
    state_values = [0]*env.n_states
    for i in range(env.n_states):
        state_values[i] = max(action_values[i,:])
    return state_values

