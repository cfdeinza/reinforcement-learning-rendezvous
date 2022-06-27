# Notes: Reinforcement meta-learning

## Gaudet2020a: (6dof asteroid landing)
Different environmental dynamics are treated as a range of POMDPs. 
In each POMDP, the policy's recurrent network hidden state will evolve differently over the course of an episode, 
capturing information regarding hidden variables (such as external forces, changes in dynamics and sensor bias). 

Even though the policy's parameters are fixed after optimization, the policy's hidden state will evolve based off the 
current POMDP, thus adapting to the environment.

After training, although the recurrent policy's network weights are frozen, the hidden state will continue to evolve 
in response to a sequence of observations and actions, thus making the policy adaptive. In contrast, an MLP policy's 
behavior is fixed by the network parameters at test time.