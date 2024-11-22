import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class VM:
    def __init__(self, cpu_demand):
        self.cpu_demand = cpu_demand
        self.current_cpu_usage = 0 # Initial CPU usage

    def monitor_cpu(self):
        return random.randint(0,100)  # Return current CPU usage for monitoring

class Server:
    def __init__(self, available_cpu):
        self.available_cpu = available_cpu
        self.vms = []  # List to store VMs allocated to the server

    def has_available_resources(self, cpu_demand):
        return self.available_cpu >= cpu_demand

    def allocate_cpu(self, cpu_demand):
        if self.has_available_resources(cpu_demand):
            self.available_cpu -= cpu_demand
            return True
        return False

    def deallocate_cpu(self, cpu_demand):
        self.available_cpu += cpu_demand

class SubstrateNetwork:
    def __init__(self, num_servers, num_switches):
        self.servers = [Server(random.randint(8,128)) for _ in range(num_servers)]
        self.switches = [None for _ in range(num_switches)]

    def find_available_server(self, cpu_demand):
        for server in self.servers:
            if server.has_available_resources(cpu_demand):
                return server
        return None

class VMRequest:
    def __init__(self, num_vms, vm_cpu_demand, bandwidth_demand):
        self.num_vms = num_vms
        self.vm_cpu_demand = vm_cpu_demand
        self.bandwidth_demand = bandwidth_demand

class VMPlacementEnv(gym.Env):
    def __init__(self, substrate_network, user_vm_requests):
        super(VMPlacementEnv, self).__init__()
        self.substrate_network = substrate_network
        self.user_vm_requests = user_vm_requests
        self.current_request = 0
        self.done = False
        self.action_space = gym.spaces.Discrete(3)  # Actions: 0 - Deallocate, 1 - Allocate, 2 - No Action
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(len(self.substrate_network.servers),), dtype=np.float32)

    def reset(self):
        self.current_request = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        vm_request = self.user_vm_requests[self.current_request]
        vms, reward_from_placement = self.place_vms(vm_request)
        reward = reward_from_placement

        for vm in vms:
            cpu_utilization = vm.monitor_cpu()
            if action == 0:
                reward -= self.deallocate_resources(vm, cpu_utilization)
            elif action == 1:
                reward -= self.allocate_resources(vm, cpu_utilization)

        self.current_request += 1
        if self.current_request >= len(self.user_vm_requests):
            self.done = True

        return self.get_state(), reward, self.done, {}

    def place_vms(self, vm_request):
        vms = []
        total_reward = 0
        for _ in range(vm_request.num_vms):
            vm = VM(vm_request.vm_cpu_demand)
            server = self.substrate_network.find_available_server(vm.cpu_demand)
            if server:
                server.allocate_cpu(vm.cpu_demand)
                server.vms.append(vm)
                vms.append(vm)
            else:
                vms.append(vm)
                total_reward += 5
        return vms, total_reward

    def deallocate_resources(self, vm, cpu_utilization):
        if cpu_utilization < 20:
            for server in self.substrate_network.servers:
                if vm in server.vms:
                    server.deallocate_cpu(vm.cpu_demand - 2)
                    return 5
        return 0

    def allocate_resources(self, vm, cpu_utilization):
        if cpu_utilization > 90:
            for server in self.substrate_network.servers:
                if vm in server.vms:
                    if server.has_available_resources(vm.cpu_demand):
                        server.allocate_cpu(vm.cpu_demand + 2)
                        return +5
        return 0

    def get_state(self):
        return np.array([server.available_cpu for server in self.substrate_network.servers])

class DQNAgent:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            target = reward + (self.gamma * torch.max(self.target_model(next_state)) * (1 - done))
            target_f = self.model(state)
            target_f[action] = target
            loss = nn.MSELoss()(target_f, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Running the simulation with DRL
def run_drl_simulation():
    num_servers = 9
    num_switches = 5
    substrate_network = SubstrateNetwork(num_servers, num_switches)

    user_vm_requests = [
        VMRequest(num_vms=3, vm_cpu_demand=4, bandwidth_demand=500),
        VMRequest(num_vms=2, vm_cpu_demand=8, bandwidth_demand=700),
        VMRequest(num_vms=5, vm_cpu_demand=6, bandwidth_demand=1100),
        VMRequest(num_vms=4, vm_cpu_demand=2, bandwidth_demand=900),
        VMRequest(num_vms=7, vm_cpu_demand=16, bandwidth_demand=800),
        VMRequest(num_vms=1, vm_cpu_demand=12, bandwidth_demand=1000),
        VMRequest(num_vms=6, vm_cpu_demand=24, bandwidth_demand=1300),
    ]

    env = VMPlacementEnv(substrate_network, user_vm_requests)
    agent = DQNAgent(action_space=env.action_space.n, state_space=env.observation_space.shape[0])

    episodes = 100
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        agent.update_target_model()
        print(f"Episode {episode+1}/{episodes} Total Reward: {total_reward}")

    # Save the trained model after all episodes
    torch.save(agent.model.state_dict(), 'dqn_vm_placement_model.pth')
    print("Model saved!")

# Load the saved model for real-time evaluation
def load_trained_model():
    agent = DQNAgent(action_space=3, state_space=5)  # Example, adjust based on your environment
    agent.model.load_state_dict(torch.load('dqn_vm_placement_model.pth'))
    agent.model.eval()  # Set to evaluation mode
    return agent

if __name__ == "__main__":
    run_drl_simulation()