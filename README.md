# VM-Allocation-using-RL

Experiments and Results:
Experimental Setup:
In this experiment, the Deep Deterministic Policy Gradient (DDPG) algorithm was employed to
optimize virtual machine (VM) allocation within a simulated cloud environment. The environment
provided real-time information on resource usage and load conditions across Physical Machines
(PMs), enabling the agent to make dynamic allocation decisions. Training was conducted over 1000
episodes, each episode representing a series of allocation decisions made under varying load
conditions. The primary evaluation metric was the total reward, which reflects the balance and
efficiency of resource utilization after each allocation.
The goal was for the agent to maximize this reward, effectively learning a policy that minimizes
resource overload while optimizing distribution across PMs.
Results and Analysis:
The plot (Fig.3) below depicts the total reward obtained by the agent over 1000 episodes. Initially,
there is a sharp drop in reward, reflecting the agent’s early-stage exploratory actions, during which it
lacks the experience to make effective allocations. Over subsequent episodes, the agent quickly
adapts, achieving a more stable reward pattern. The total reward consistently centers around -5.5 after
an initial adaptation period, indicating that the agent has learned a reasonably efficient allocation
policy that it follows throughout most episodes.
Although the reward fluctuates, the overall trend stabilizes, suggesting that the agent has developed a
steady policy. However, minor variations in reward across episodes likely stem from the stochastic
nature of the environment, as VM demand and resource usage can vary unpredictably. These
fluctuations, which typically range between -5.25 and -5.75, suggest that while the agent’s policy is
effective, there may be occasional allocations that result in minor inefficiencies.
This output (Fig.2) illustrates a typical VM allocation process in a cloud data center simulation,
providing a detailed view of resource management for a single episode. Each Virtual Machine (VM)
has specific CPU and memory requirements, and the model's objective is to allocate these VMs
efficiently across Physical Machines (PMs) to optimize resource utilization and minimize energy
consumption.In this example, the allocation begins with PM 1 as the primary target due to its initial
availability. For each VM, the model checks if PM 1 has enough resources to meet the VM's CPU and
memory needs. If PM 1 has sufficient capacity, the VM is allocated to it. However, once PM 1's
available resources drop below the required threshold for a VM, the model looks for alternative PMs
with adequate remaining capacity. For instance, when PM 1 is unable to accommodate a VM, other
PMs, such as PMs 4 and 5, serve as fallback options.
The episode concludes with a summary of key performance metrics, including the total reward,
average CPU utilization, total energy consumption, and any overload penalties incurred. These
metrics provide feedback on the model's performance, with the reward indicating the effectiveness of
the allocation strategy. The model aims to achieve balanced resource utilization, reducing overloads
while maintaining energy efficiency.
