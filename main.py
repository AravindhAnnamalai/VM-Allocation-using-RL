import torch
import numpy as np
from cloud_simulation import Datacenter, PhysicalMachine, VirtualMachine
from replay_buffer import ReplayBuffer
from actor_critic import ActorNetwork, CriticNetwork
import random
import matplotlib.pyplot as plt

# Define DDPG parameters
state_size = 4
action_size = 1
TAU = 0.001
LEARNING_RATE = 0.0001
BATCH_SIZE = 64

# Initialize networks and replay buffer
actor = ActorNetwork(state_size, action_size, TAU, LEARNING_RATE)
critic = CriticNetwork(state_size, action_size, TAU, LEARNING_RATE)
replay_buffer = ReplayBuffer(10000)

# Training parameters
num_episodes = 100
rewards = []

# Reward function
def calculate_reward(pm, success, overload_penalty=0.5):
    cpu_utilization = pm.get_cpu_utilization() / pm.total_cpu  # Normalize
    memory_utilization = pm.get_memory_utilization() / pm.total_memory  # Normalize
    energy_consumption = pm.get_energy_consumption()
    reward = (cpu_utilization + memory_utilization) - (0.1 * energy_consumption + overload_penalty)
    return reward if success else reward - overload_penalty

# Training loop
for episode in range(num_episodes):
    # Reinitialize datacenter and VMs for each episode
    datacenter = Datacenter()
    datacenter.add_pm(PhysicalMachine(1, total_cpu=16, total_memory=32))
    datacenter.add_pm(PhysicalMachine(2, total_cpu=8, total_memory=16))
    datacenter.add_pm(PhysicalMachine(3, total_cpu=32, total_memory=64))
    datacenter.add_pm(PhysicalMachine(4, total_cpu=16, total_memory=32))
    datacenter.add_pm(PhysicalMachine(5, total_cpu=64, total_memory=128))
    
    # Define VMs with reduced resource requirements
    vms = [VirtualMachine(i, cpu_required=2 + i // 2, memory_required=4 + i) for i in range(1, 11)]
    for vm in vms:
        datacenter.add_vm(vm)

    total_reward = 0
    total_cpu_utilization = 0
    total_memory_utilization = 0
    total_energy_consumption = 0
    total_overload_penalty = 0

    print(f"\nEpisode {episode + 1}")

    for vm in datacenter.vms:
        # Display VM details
        print(f"\nAttempting to allocate VM {vm.vm_id} (CPU: {vm.cpu_required}, Memory: {vm.memory_required})")

        # State for actor network
        state = torch.FloatTensor([datacenter.pms[0].available_cpu, datacenter.pms[0].available_memory, vm.cpu_required, vm.memory_required]).unsqueeze(0)
        action = actor(state).detach().numpy() + np.random.normal(0, 0.1, action_size)  # Exploration noise
        action = torch.tensor(action).float()

        # Map action to PM index across all PMs
        pm_index = int((action[0] + 1) / 2 * len(datacenter.pms))
        pm_index = max(0, min(pm_index, len(datacenter.pms) - 1))
        pm = datacenter.pms[pm_index]

        # Attempt initial allocation to chosen PM
        print(f"Selected PM {pm.pm_id} with Available CPU: {pm.available_cpu}, Memory: {pm.available_memory}")
        success = pm.allocate_vm(vm)

        # If allocation fails, use fallback allocation to distribute to another PM
        if not success:
            print(f"Failed to allocate VM {vm.vm_id} to PM {pm.pm_id}. Attempting alternative PMs...")

            # Try other PMs until allocation succeeds or all fail
            for alternative_pm in random.sample(datacenter.pms, len(datacenter.pms)):
                if alternative_pm.allocate_vm(vm):
                    print(f"VM {vm.vm_id} allocated to PM {alternative_pm.pm_id} as fallback.")
                    pm = alternative_pm
                    success = True
                    break

            if not success:
                print(f"VM {vm.vm_id} failed to allocate to any PM.")

        # Calculate reward and accumulate metrics
        reward = calculate_reward(pm, success)
        total_reward += reward
        next_state = torch.FloatTensor([pm.available_cpu, pm.available_memory, vm.cpu_required, vm.memory_required]).unsqueeze(0)
        replay_buffer.add(state.numpy(), action.numpy(), reward, next_state.numpy(), success)

        total_cpu_utilization += pm.get_cpu_utilization()
        total_memory_utilization += pm.get_memory_utilization()
        total_energy_consumption += pm.get_energy_consumption()
        if not success:
            total_overload_penalty += 0.5

        # Replay buffer training
        if replay_buffer.size() > BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            for s, a, r, s2, d in batch:
                state = torch.FloatTensor(s)
                action = torch.FloatTensor(a)
                reward = torch.FloatTensor([r])
                next_state = torch.FloatTensor(s2)

                # Target actions and future rewards
                target_action = actor.target_model(next_state).detach()
                future_reward = critic.target_forward(next_state, target_action).detach()
                q_value = reward if d else reward + 0.99 * future_reward
                q_value = q_value.view(1, 1)

                # Train critic
                critic.train_critic(state, action, q_value)

                # Train actor
                predicted_action = actor(state)
                actor_loss = -critic(state, predicted_action).mean()
                actor.optimizer.zero_grad()
                actor_loss.backward()
                actor.optimizer.step()

                # Update target networks
                actor.target_train()
                critic.target_train()

    # Record episode metrics
    rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    print(f"  Total CPU Utilization: {total_cpu_utilization}")
    print(f"  Total Memory Utilization: {total_memory_utilization}")
    print(f"  Total Energy Consumption: {total_energy_consumption}")
    print(f"  Total Overload Penalty: {total_overload_penalty}\n")

# Plot Reward vs Number of Episodes
plt.plot(range(1, num_episodes + 1), rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward vs Number of Episodes")
plt.show()
