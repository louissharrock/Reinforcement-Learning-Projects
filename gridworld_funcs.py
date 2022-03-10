# import packages
import gym
import numpy as np
import random
import copy

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
def simulate_episode(env, policy, n_moves=5,print_moves=False):
    env = gym.make("GridWorld-v0")
    env.reset()
    if(print_moves):
        print("The initial state:")
        env.render()
    total_reward = 0
    episode = []
    for i in range(n_moves):
        state = env.state
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
    counter = 0
    while counter < n_sweeps:
        for s in range(env.n_states):
            state = env.ind_to_state(s)
            values[s] = evaluate_single_policy(env, values, policy, state, discount_rate, deterministic=deterministic)
        counter += 1
    return values


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
    is_policy_stable = False
    counter = 0
    while is_policy_stable is False:
        counter += 1
        current_values = evaluate_all_policy(env, values, current_policy, discount_rate, n_eval)
        current_policy, is_policy_stable = improve_all_policy(env, current_values, current_policy, discount_rate)
    return current_policy, current_values, counter
