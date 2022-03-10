from gridworld_funcs import *

# For reproducibility
random.seed(1)

# Create environment
env = gym.make("GridWorld-v0", nrow=5, ncol=5)
env.reset()


# Create a randomly chosen deterministic policy
policy0 = get_policy(env)
#print("\nThe randomly chosen deterministic policy (POLICY0) is")
#print(policy0)

# Create a uniform (equal probabilities) random policy
policy1 = get_uniform_policy(env)
#print("\nThe uniform (i.e. equal probabilities) policy (POLICY1) is")
#print(policy1, "\n")


# Compute action under a given policy for a single state
state0 = np.array([0, 0])
action_policy0_state0 = get_action(env, policy0, state0)
action_policy1_state0 = get_action(env, policy1, state0)
#print("\nThe action (deterministically) chosen for the state", list(state0), "according to POLICY0 is", action_policy0_state0)
#print("The action (randomly) chosen for the state", list(state0), "according to POLICY1 is", action_policy1_state0, "\n")


# Compute actions under a given policy for all states
input("First let's take a look at the actions chosen under two policies. Press enter to continue.")
print()

print("The actions chosen according to the deterministic POLICY0 are:")
print_actions(env, policy0)

print("The actions chosen according to the random POLICY1 are:")
print_actions(env, policy1)


# Simulate episode under these policies
input("Next let's simulate an episode under each of these policies. Press enter to continue.")
print()
n_moves = 10

episode_policy0, total_reward_policy0 = simulate_episode(env,policy0,n_moves)
print("The (state,action,reward) triplets for the episode under POLICY0 were:")
print_episode(episode_policy0,total_reward_policy0)
print()

episode_policy1, total_reward_policy1 = simulate_episode(env,policy1,n_moves)
print("The (state,action,reward) triplets for the episode under POLICY1 were:")
print_episode(episode_policy1,total_reward_policy1)
print()

# Compute value function of a single state under a given policy
discount_rate = 0.9
values_init = [0] * env.n_states

value_policy0_state0 = evaluate_single_policy(env, values_init, policy0, state0, discount_rate)
#print("The value of state", state0, "under POLICY0 is", value_policy0_state0, "\n")

value_policy1_state0 = evaluate_single_policy(env, values_init, policy1, state0, discount_rate, deterministic=False)
#print("The value of state", state0, "under POLICY1 is", value_policy1_state0, "\n")


# Compute value function of all states under a given policy
n_sweeps = 50

input("Now let's compute the value function under each of the two policies. Press enter to continue...")

print()
discount_rate = 0
values_policy0 = evaluate_all_policy(env, values_init, policy0, discount_rate, n_sweeps)
print("The value of the states under POLICY0, with discount factor 0, is given by")
print_values(env, values_policy0)

discount_rate = 0.9
values_policy1 = evaluate_all_policy(env, values_init, policy1, discount_rate, n_sweeps, deterministic=False)
print("The value of the states under POLICY1, with discount factor 0.9, is given by") # compare with Figure 3.5 in Sutton
print_values(env, values_policy1)


# Compute improved policy for a single state
action_policy0_dash_state0 = improve_single_policy(env, values_init, policy0, state0, discount_rate)
#print("The original policy in", state0, "was to move", action_policy0_state0)
#print("The improved policy in", state0, "is to move", action_policy0_dash_state0, "\n")


# Compute improved policy for all states
input("Next let's compute an improvement to POLICY0. Press enter to continue...")
print()

action_policy0_dash, is_policy_stable = improve_all_policy(env,values_init,policy0,discount_rate)

print("The original policy was")
print_actions(env,policy0)

print("The improved policy is")
print_actions(env,action_policy0_dash)


# Perform policy evaluation and policy iteration
input("Finally, we can compute the optimal policy using dynamic programming. Press enter to continue...")
print()

policy_opt, values_opt, counter = dynamic_program(env,values_init,policy0,discount_rate,n_sweeps)
print("The optimal policy is")
print_actions(env, policy_opt)
#print("The optimal value function is")
#print_values(env,values_opt)
#print("The number of iterations (policy improvements) to obtain the optimal policy was", counter)