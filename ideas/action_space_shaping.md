# Action space shaping

We can improve the learning process by shaping the action space, much like we shape a reward function. 

### *Action Space Shaping in Deep Reinforcement Learning* by Kannervisto et al

Refers mostly to video-game applications. Identifies 3 main methods to shape the action space:
- Remove actions: reducing the number of actions helps with exploration because there are less actions to try. 
This in turn improves the sample efficiency of the training. However, it requires domain knowledge, and may restrict the agent's capabilities.
- Discretize continuous actions: continuous actions can be discretized by splitting them into a set of bins, or by defining three discrete choices (negative, zero, positive)
- Convert multi-discrete actions to discrete

They ran experiments with PPO (SB3 and rllib) and found that removing actions and discretizing continuous actions improved learning. 
Converting multi-discrete to discrete did not have a clear positive effect.

### *Policy invariance under reward transformations* by Andrew Ng et al

Investigates which modifications to the reward function preserve the optimal policy:
- positive linear transformation
- addition of a reward for transitions between states expressed as a potential function

Any other modification may yield suboptimal policies. 
Also describes methods to construct shaped rewards corresponding to distance-based and subgoal-based heuristics. 
Also shows that non-potential-based rewards can lead to "bugs".

Ideas:
- Give a positive reward whenever the agent has moved closer to the goal, and zero otherwise.
- To encourage taking an action *a1* at state *s*, give a positive reward whenever *a=a1* at state *s*, zero otherwise.

If there is some sequence of states such that the agent can travel through them in a cycle and gain a net positive reward, 
then the agent may get "distracted" from whatever it needs to be doing, and instead try to repeatedly go around this cycle. 
To address this difficulty with cycles, a form of reward shaping that immediately comes to mind is the *difference of potentials*.

$ R = \phi (s') - \phi (s) $

where \phi is some function over states, and *s'* is the state after *s*.

If expert knowledge about the domain is available, then non-potential shaping functions might also be fully appropriate.