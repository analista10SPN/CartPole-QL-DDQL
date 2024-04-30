import numpy as np
import gymnasium as gym

def discretize_state(state, bins):
    """Convert continuous state into a discretized state."""
    discretized_state = []
    for dim, val in enumerate(state):
        bin_index = np.digitize(val, bins[dim]) - 1
        bin_index = max(0, min(bin_index, len(bins[dim]) - 2))  # Keep within bounds
        discretized_state.append(bin_index)
    return tuple(discretized_state)

def reward_function(x, theta):
    """Custom reward function based on system requirements."""
    # Normalize theta within [-pi, pi] for calculations
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    reward = -(x**2 + (theta - np.pi)**2)  # Penalize deviation from desired state
    return reward

def q_learning(episodes=500000, bins=None, alpha=0.1, gamma=0.99, epsilon=1.0, decay=0.99995):
    env = gym.make('CartPole-v1')
    if bins is None:
        bins = [
            np.linspace(-2.4, 2.4, 6),  # Cart position
            np.linspace(-3.0, 3.0, 12),  # Cart velocity
            np.linspace(-np.pi, np.pi, 12),  # Pole angle sin
            np.linspace(-np.pi, np.pi, 12)   # Pole angle cos
        ]

    Q = np.zeros([len(b) + 1 for b in bins] + [env.action_space.n])
    rewards = []

    for e in range(episodes):
        state = discretize_state(env.reset()[0], bins)
        total_reward = 0
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            
            info = env.step(action)
            # print("INFO:",info)
            next_state = info[0]
            # print("action from env.step(action)",next_state)
            reward = info[1] 
            done = info[2]
            reward = reward_function(next_state[0], next_state[2])  # x and theta

            next_state = discretize_state(next_state, bins)
            next_max = np.max(Q[next_state])
            Q[state + (action,)] += alpha * (reward + gamma * next_max - Q[state + (action,)])

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon *= decay  # Reduce epsilon to decrease exploration over time

        if (e + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode: {e + 1}, Average Reward: {avg_reward}, Epsilon: {epsilon}")

    env.close()
    return Q

# Run the Q-learning algorithm
q_table = q_learning()
