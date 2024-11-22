import matplotlib.pyplot as plt

# Function to plot the CPU utilization across all servers
def plot_cpu_utilization(substrate_network):
    server_ids = range(len(substrate_network.servers))
    cpu_utilization = [server.available_cpu for server in substrate_network.servers]

    plt.figure(figsize=(10, 6))
    plt.bar(server_ids, cpu_utilization, color='blue')
    plt.xlabel("Server ID")
    plt.ylabel("Available CPU (%)")
    plt.title("CPU Utilization Across Servers")
    plt.show()

# Function to plot total reward over time
def plot_total_reward(total_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards, color='green')
    plt.xlabel("Episode/Request")
    plt.ylabel("Total Reward")
    plt.title("Total Reward During Evaluation")
    plt.show()

# Function to plot the distribution of actions taken (Allocate, Deallocate, No Action)
def plot_action_distribution(action_counts):
    actions = ['Allocate', 'Deallocate', 'No Action']
    plt.figure(figsize=(10, 6))
    plt.bar(actions, action_counts, color='purple')
    plt.xlabel("Action Type")
    plt.ylabel("Frequency")
    plt.title("Distribution of Actions Taken by Agent")
    plt.show()

# Modify the evaluate_trained_model function to track these metrics
def evaluate_trained_model():
    agent = load_trained_model()

    num_servers = 15
    num_switches = 8
    substrate_network = SubstrateNetwork(num_servers, num_switches)

    user_vm_requests = [
        VMRequest(num_vms=3, vm_cpu_demand=4, bandwidth_demand=500),
        VMRequest(num_vms=2, vm_cpu_demand=8, bandwidth_demand=700),
        VMRequest(num_vms=5, vm_cpu_demand=6, bandwidth_demand=1100),
        VMRequest(num_vms=4, vm_cpu_demand=10, bandwidth_demand=900),
        VMRequest(num_vms=7, vm_cpu_demand=16, bandwidth_demand=800),
        VMRequest(num_vms=1, vm_cpu_demand=32, bandwidth_demand=1000),
        VMRequest(num_vms=6, vm_cpu_demand=24, bandwidth_demand=1600),
        VMRequest(num_vms=4, vm_cpu_demand=10, bandwidth_demand=900),
        VMRequest(num_vms=1, vm_cpu_demand=32, bandwidth_demand=1000),
        VMRequest(num_vms=7, vm_cpu_demand=16, bandwidth_demand=800),
        VMRequest(num_vms=3, vm_cpu_demand=4, bandwidth_demand=500)
    ]

    env = VMPlacementEnv(substrate_network, user_vm_requests)

    total_rewards = []  # List to track total rewards
    action_counts = [0, 0, 0]  # [Allocate, Deallocate, No Action]

    for vm_request in user_vm_requests:
        state = env.reset()  # Reset the environment
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)  # Get the action from the trained agent
            next_state, reward, done, _ = env.step(action)  # Take the action in the environment
            total_reward += reward  # Accumulate the reward
            action_counts[action] += 1  # Track the action taken
            state = next_state  # Move to the next state

        total_rewards.append(total_reward)

    # After the evaluation, plot the metrics
    plot_cpu_utilization(substrate_network)
    plot_total_reward(total_rewards)
    plot_action_distribution(action_counts)

# Load the trained model for evaluation
def load_trained_model():
    # Initialize the agent with the correct state space size (15) as used in training
    agent = DQNAgent(action_space=3, state_space=9)  # Example, adjust based on your environment
    agent.model.load_state_dict(torch.load('dqn_vm_placement_model.pth'))
    agent.model.eval()  # Set to evaluation mode
    return agent


# Run evaluation and plotting
if __name__ == "__main__":
    evaluate_trained_model()  # Evaluate the trained model and plot the results
