# Implementation details for PPO

Information taken from [blogpost](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) and 
[sb3-contrib](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html).

There are two main phases: the rollout phase and the learning phase. 
During the rollout phase, the agent collects experience by interacting with the environment.
During the learning phase, the agent from the collected data. Specifically, it estimates the values of the observations, 
calculates the advantages, computes the clipped surrogate loss and the value loss, and optimizes the networks.

## Vectorized architecture:
Take advantage of the efficient paradigm known as *vectorized architecture*, which features a learner that collects samples from multiple environments. 
(i.e. it always receives N outputs from N environments, and selects N actions to step the N environments)

## LSTM:
Users are advised to start with [frame-stacking](https://wandb.ai/sb3/no-vel-envs/reports/PPO-vs-RecurrentPPO-aka-PPO-LSTM-on-environments-with-masked-velocity--VmlldzoxOTI4NjE4), 
which is simpler, faster, and usually competitive. 
You can stack multiple observations using `VecFrameStack`.

### Available policies:
- `MlpLstmPolicy` (alias of `RecurrentActorCriticPolicy`)
- `CnnLstmPolicy` (alias of `RecurrentActorCriticCnnPolicy`)
- `MultiInputLstmPolicy` (alias of `RecurrentMultiInputActorCriticPolicy`)

(Remember that in SB3 the term "policy" refers to the class than handles all the networks during training, including actor and critic)

The action space and observation space can take any form (Discrete, Box, MultiDiscrete, MultiBinary) except Dict (for the action space).

It is particularly important to pass the `lstm_states` and `episode_start` argument to the `predict()` method, 
so that the cell and hidden states of the LSTM are correctly updated.

I downloaded the required package: `pip install sb3-contrib` (I think this also updated my sb3 package from 1.5.0 to 1.6.0)

It seems easy enough to implement during training. Simply use the `RecurrentPPO` model instead of the `PPO` model, and the 
`MlpLstmPolicy` instead of the `MlpPolicy`. However, I will have to make some changes to the evaluation script, because 
I need to pass additional arguments to the `predict()` method, as mentioned above. 
Also I don't know if the `evaluate_policy()` method works properly for the RNN model 
(`evaluate_policy()` is used in the callback during training, so if it does not work properly then we are not saving the best model).

## Custom architectures:
The architecture of a network is given by the `net_arch` parameter.

One way to customize the policy network architecture is to pass the `policy_kwargs` argument when creating the model. 
For example: custom actor (pi) and value function (vf) networks of two layers with 32 neurons each, and ReLU activation function
```python
import torch as th
from stable_baselines3 import PPO

policy_kwargs = dict(
    activation_fn=th.nn.ReLU, 
    net_arch=[dict(pi=[32, 32], vf=[32, 32])]
)

model = PPO("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs)

```

### Shared networks:
The `net_arch` parameter can also be used to specify layers that are shared by the policy network and the value network. 
The format is:
```
net_arch = [<shared_layers>, dict(vf=[<non-shared value network layers>], pi=[<non-shared policy network layers>])]
```
For example:
- Two shared layers of 128 neurons: `net_arch=[128, 128]`
- Initially shared, then diverging: `net_arch=[128, dict(vf=[64, 64], pi=[64])]`

### Advanced:
If you need even more control over the policy, you can define a [custom policy class](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#advanced-example) 
which inherits from `ActorCriticPolicy`.

## Recurrent:
`RecurrentActorCriticPolicy` (see [source code](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/7993b75781d7f43262c80c023cd83cfe975afe3a/sb3_contrib/common/recurrent/policies.py#L22))
 inherits from `ActorCriticPolicy` (which is just a MLP). It applies a multi-layer LSTM to the MLP. 
 The default number of LSTM layers is 1, and the default number of features in the hidden state is 256. By default, 
 the actor and the critic do not share the same LSTM.