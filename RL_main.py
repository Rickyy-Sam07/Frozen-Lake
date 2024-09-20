import numpy as np
import time
import gym
import os

# Create the environment
env = gym.make('FrozenLake-v1')

# Assuming q_table is already defined or load it from a file if necessary
# For demonstration, initialize a random q_table
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.random.rand(state_space_size, action_space_size)

max_steps_per_episode = 100  # Define your max steps per episode

def clear_console():
    # Clear the console based on the operating system
    os.system('cls' if os.name == 'nt' else 'clear')

def render_frozenlake(env, state):
    # Custom function to render the FrozenLake environment
    desc = env.unwrapped.desc
    desc = [[c.decode('utf-8') for c in line] for line in desc]
    row, col = divmod(state, env.unwrapped.ncol)
    desc[row][col] = 'A'  # Mark the agent's position
    return "\n".join("".join(line) for line in desc)

action_labels = ['Left', 'Down', 'Right', 'Up']

for episode in range(3):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # Handle the case where reset returns a tuple
    done = False
    print("CHAPTER", episode + 1, "\n\n\n\n")
    time.sleep(1)
    for step in range(max_steps_per_episode):
        clear_console()
        print(render_frozenlake(env, state))  # Render the environment using the custom function
        action = np.argmax(q_table[state, :])
        print(f"Action taken: {action_labels[action]}")
        time.sleep(0.5)  # Adjust time delay for better visualization

        # Take the action
        result = env.step(action)
        if len(result) == 5:
            new_state, reward, done, truncated, info = result
            done = done or truncated  # Combine done and truncated for newer gym versions
        else:
            new_state, reward, done, info = result

        if done:
            clear_console()
            print(render_frozenlake(env, new_state))
            if reward == 1:
                print("SUCCESS")
            else:
                print("FAILURE")
            time.sleep(3)
            clear_console()
            break

        state = new_state

env.close()
