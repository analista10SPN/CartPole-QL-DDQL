import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

no_buckets = (1, 1, 6, 3)
no_actions = env.action_space.n

state_value_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_value_bounds[1] = (-0.5, 0.5)
state_value_bounds[3] = (-math.radians(50), math.radians(50))

print(state_value_bounds)
print(len(state_value_bounds))
print(np.shape(state_value_bounds))
print(state_value_bounds[0][0])

# define q_value_table
q_value_table = np.zeros(no_buckets + (no_actions,))

# Q has 6 dimensions 1 x 1 x 6 x 3 x 2
print(q_value_table)


# user-defined parameters
min_explore_rate = 0.1
min_learning_rate = 0.1
max_episodes = 1000
max_time_steps = 250
streak_to_end = 120
solved_time = 199
discount = 0.99
no_streaks = 0

energy_expenditure_per_episode = []
convergence_times = []

total_energy_expenditure = 0
first_convergence_episode = None

mass_cart = 1.0  # Mass of the cart
mass_pole = 0.1  # Mass of the pole
gravity = 9.8    # Gravitational acceleration
pole_length = 0.5 # Half the pole's length

def calculate_energy(observation):
    x, x_dot, theta, theta_dot = observation
    # Kinetic energy of the cart
    kinetic_energy_cart = 0.5 * mass_cart * x_dot**2
    # Kinetic energy of the pole (rotational)
    kinetic_energy_pole = 0.5 * (1/3) * mass_pole * pole_length**2 * theta_dot**2
    # Potential energy of the pole (assuming the bottom of the pole as reference point)
    potential_energy_pole = mass_pole * gravity * (pole_length * (1 - np.cos(theta)))
    # Total energy
    total_energy = kinetic_energy_cart + kinetic_energy_pole + potential_energy_pole
    return total_energy

# Select an action - explore vs exploit
# epsilon-greedy method
def select_action(state_value, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()  # explore
    else:
        action = np.argmax(q_value_table[state_value])  # exploit
    return action


def select_explore_rate(x):
    # change the exploration rate over time.
    return max(min_learning_rate, min(1.0, 1.0 - math.log10((x + 1) / 25)))


def select_learning_rate(x):
    # Change learning rate over time
    return max(min_learning_rate, min(1.0, 1.0 - math.log10((x + 1) / 25)))


def bucketize_state_value(state_value):
    """ Discretizes continuous values into fixed buckets"""
    state_value = np.array(state_value)  # Ensure it's a numpy array for easier manipulation
    bucket_indices = []
    for i in range(len(state_value)):
        current_value = state_value[i]
        # Ensure current_value is a scalar, not an array
        if isinstance(current_value, np.ndarray):
            current_value = current_value.item()  # Get the Python scalar value

        if current_value <= state_value_bounds[i][0]:  # violates lower bound
            bucket_index = 0
        elif current_value >= state_value_bounds[i][1]:  # violates upper bound
            bucket_index = no_buckets[i] - 1  # put in the last bucket
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (no_buckets[i] - 1) * state_value_bounds[i][0] / bound_width
            scaling = (no_buckets[i] - 1) / bound_width
            bucket_index = int(round(scaling * current_value - offset))

        bucket_indices.append(bucket_index)
    return tuple(bucket_indices)


# main program
if __name__ == "__main__":

    _DEBUG = True
    frames = []
    reward_per_episode = []
    time_per_episode = []
    avgtime_per_episode = []
    learning_rate_per_episode = []
    explore_rate_per_episode = []

    # train the system
    totaltime = 0
    for episode_no in range(max_episodes):
        episode_energy_expenditure = 0  # Reset energy expenditure for the episode
        

        explore_rate = select_explore_rate(episode_no)
        learning_rate = select_learning_rate(episode_no)

        learning_rate_per_episode.append(learning_rate)
        explore_rate_per_episode.append(explore_rate)

        # reset the environment while starting a new episode
        observation, info = env.reset()
        previous_energy = calculate_energy(observation)
        # print("Observation:", observation)
        # print("Info:", info)
        

        start_state_value = bucketize_state_value(observation)
        previous_state_value = start_state_value

        done = False
        time_step = 0

        while not done:
            # env.render()
            action = select_action(previous_state_value, explore_rate)
            step_results = env.step(action)
            # print("Step results:", step_results) 
            observation, reward_gain, done,_, info = step_results
            current_energy = calculate_energy(observation)
            energy_difference = abs(current_energy - previous_energy)
            episode_energy_expenditure += energy_difference
            previous_energy = current_energy
            
            
            
            state_value = bucketize_state_value(observation)
            best_q_value = np.max(q_value_table[state_value])

            # update q_value_table
            q_value_table[previous_state_value][action] += learning_rate * (
                reward_gain
                + discount * best_q_value
                - q_value_table[previous_state_value][action]
            )

            previous_state_value = state_value

            if episode_no % 100 == 0 and _DEBUG is True:
                print("Episode number: {}".format(episode_no))
                print("Time step: {}".format(time_step))
                print("Previous State Value: {}".format(previous_state_value))
                print("Selected Action: {}".format(action))
                print("Current State: {}".format(str(state_value)))
                print("Reward Obtained: {}".format(reward_gain))
                print("Best Q Value: {}".format(best_q_value))
                print("Learning rate: {}".format(learning_rate))
                print("Explore rate: {}".format(explore_rate))

            time_step += 1
            # while loop ends here

        energy_expenditure_per_episode.append(episode_energy_expenditure)
        if time_step >= solved_time:
            no_streaks += 1
        else:
            no_streaks = 0

        if no_streaks > streak_to_end:
            print("CartPole problem is solved after {} episodes.".format(episode_no))
            break

        # data log
        if episode_no % 100 == 0:
            print(
                "Episode {} finished after {} time steps".format(episode_no, time_step)
            )
        time_per_episode.append(time_step)
        totaltime += time_step
        avgtime_per_episode.append(totaltime / (episode_no + 1))
        # episode loop ends here

    env.close()

    # Plotting
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(time_per_episode)
    axes[0].plot(avgtime_per_episode)
    axes[0].set(ylabel="time per episode")
    axes[1].plot(learning_rate_per_episode)
    axes[1].plot(explore_rate_per_episode)
    axes[1].set_ylim([0, 1])
    axes[1].set(xlabel="Episodes", ylabel="Learning rate")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(energy_expenditure_per_episode)
    plt.title('Energy Expenditure per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Energy Expenditure')
    plt.grid(True)
    plt.show()

    # Show final result
    observation = env.reset()

    start_state_value = bucketize_state_value(observation)
    previous_state_value = start_state_value

    done = False

    while not done:
        env.render()
        action = select_action(previous_state_value, explore_rate)
        observation, reward_gain, done,_, info = env.step(action)
        state_value = bucketize_state_value(observation)
        previous_state_value = state_value