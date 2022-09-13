# Action space shaping

We can improve the learning process by shaping the action space, much like we shape a reward function. 

### *Action Space Shaping in Deep Reinforcement Learning* by Kannervisto

Refers mostly to video-game applications. Identifies 3 main methods to shape the action space:
- Remove actions: reducing the number of actions helps with exploration because there are less actions to try. 
This in turn improves the sample efficiency of the training. However, it requires domain knowledge, and may restrict the agent's capabilities.
- Discretize continuous actions: continuous actions can be discretized by splitting them into a set of bins, or by defining three discrete choices (negative, zero, positive)
- Convert multi-discrete actions to discrete

They ran experiments with PPO (SB3 and rllib) and found that removing actions and discretizing continuous actions improved learning. 
Converting multi-discrete to discrete did not have a clear positive effect.