# Notes: Reinforcement meta-learning

## Gaudet2020a: (6dof asteroid landing)
### Intro:
Different environmental dynamics are treated as a range of POMDPs. 
In each POMDP, the policy's recurrent network hidden state will evolve differently over the course of an episode, 
capturing information regarding hidden variables (such as external forces, changes in dynamics and sensor bias). 

Even though the policy's parameters are fixed after optimization, the policy's hidden state will evolve based off the 
current POMDP, thus adapting to the environment.

After training, although the recurrent policy's network weights are frozen, the hidden state will continue to evolve 
in response to a sequence of observations and actions, thus making the policy adaptive. In contrast, an MLP policy's 
behavior is fixed by the network parameters at test time.

### POMDP:
A Partially Observable Markov Decision Process (POMDP) is a Markov Decision Process (MDP) where the agent cannot 
directly observe the underlying state of the system. At every time step, the agent receives an observation $o$ instead of the state of the system $x$.
In other words, the state $x$ becomes a hidden state, and we use an observation function $O(x)$ to map states to observations.

### Meta-learning:
With reinforcement meta-learning, the agent learns to quickly adapt to novel POMDPs by learning over a wide range of POPMDPs. 
The POMDPs can have different parameters, such as:
- different environmental dynamics
- actuator failure scenarios
- mass and inertial tensor variation
- sensor distortion

An agent can learn to adapt to these uncertain conditions, often after only a few steps of interaction with the environment.

There are different ways to implement meta-learning, including:
- design the objective function(?) to make the model parameters transfer well to new tasks [(2017)](https://arxiv.org/abs/1703.03400)
- use a hierarchy of policies [(2017)](https://arxiv.org/abs/1710.09767).

The authors of this work use PPO with recurrent layers in both the policy and the value networks. During training, the 
hidden state of a network's recurrent layer evolves differently depending on the observed sequence of observations and actions. 
The hidden state captures unobserved (potentially time-varying) information such as external forces that are useful in minimizing the cost function. 
In contrast, a non-recurrent policy can only optimize using a set of current observations, and will tend to underperform on tasks with high uncertainty 
(although it can still give good results if the uncertainties are not too extreme).

After training, even though the network parameters are frozen, the hidden state continues to evolve in response to a sequence of observations and actions, 
thus making the policy adaptive. In contrast, the behavior of a non-recurrent policy is fixed.

The structure of the policy and value networks is almost identical (only differs in the number of neurons in some layers). 
They have three hidden layers with *tanh* activation functions. The second layer is recurrent, implemented as a [gated recurrent unit](http://proceedings.mlr.press/v37/chung15.html).

## Federici2022:
Uses the [Ray-RL](https://docs.ray.io/en/master/rllib/index.html) reinforcement learning library.

The reinforcement meta-learning approach that uses a recurrent neural network was inspired by 
[Wang et al](https://arxiv.org/pdf/1611.05763.pdf) "Learning to reinforcement learning" in 2016.

RNNs are a particular type of RNNs that can keep track of the temporal variation of the observations collected during training. 
This capability significantly boosts the average performance achieved by RL in complex environments such as:
- non-Markov environments
- multiple task environments
- unknown or partially-observable environments
The versatility of meta-RL has been confirmed by a number of works that studied asteroid close-proximity operations, landing, and intercept missions.

The input to the policy network is an artifially-generated image (using Blender), so it has 4 convolutional filters. 
The output of the convolutional block is fed into a multi-layer perceptron (MLP) with two hidden layers. 
Lastly, the output of the MLP is fed to a Long Short-Term Memory (LSTM) block (it is a recurrent layer capable of understanding the temporal relationship between the observations).

Image inputs are a good use-case for meta-RL, because the image provides incomplete information of the system (it cannot show velocity or motion).

A different approach for learning the temporal relationships between observations is to stack a number of images together as one input. 
(Used on [Atari problems](https://doi.org/10.1109/SCC49971.2021), 2021).

Future investigations will consider the presence of other perturbative accelerations, uncertain initial conditions
of the spacecraft, noisy images and control errors. The spacecraft attitude will be considered as well as part of the controllable dynamics. Tests with a different resolution of the camera onboard will be carried out too