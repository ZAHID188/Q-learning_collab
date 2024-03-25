import numpy as np

# Define the Q-table
num_states = 5
num_actions = 3
Q = np.zeros((num_states, num_actions))
print(Q)

# Define the rewards matrix
rewards = np.array([
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [-1, -1, -1],
    [0, -1, 100]
])
print(rewards)
# Define the discount factor
gamma = 0.8

# Define the learning rate
learning_rate = 0.5

# Define the number of episodes
num_episodes = 1000

# Q-learning algorithm
for episode in range(num_episodes):
    state = np.random.randint(0, num_states)  # Initial state
    done = False
    
    while not done:
        action = np.argmax(Q[state])  # Choose action with highest Q-value
        
        next_state = np.random.randint(0, num_states)  # Randomly select next state
        reward = rewards[state, action]  # Get reward for current state-action pair
        
        Q[state, action] += learning_rate * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        state = next_state  # Move to next state
        
        if state == num_states - 1:
            done = True

# Testing the learned policy
state = 0  # Initial state

while state != num_states - 1:
    action = np.argmax(Q[state])  # Choose action with highest Q-value
    
    print("Current state:", state)
    print("Action:", action)
    
    state = np.random.randint(0, num_states)  # Randomly select next state

print("Reached the goal stat!e!")