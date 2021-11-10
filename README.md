#Basic info:
The goal of this repository is to train an agent to perform an autonomous rendezvous in a predefined environment. 
The agent is trained with a reinforcement learning algorithm from Stable-Baselines3.

## Contents:
- `custom_callbacks.py`: contains callback functions used during training.\
- `custom_env.py`: contains environments created according to the guidelines of OpenAI Gym.\
- `main.py`: the main script that is executed to train or evaluate a model.\
- `plot_*.py`: these scripts plot the trajectories of the different models.

## Stable-Baselines 3:
[Stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) is an open source project of DLR's Institute of Robotics and Mechatronics. 
It provides a library of reliable reinforcement learning algorithms implemented in PyTorch. 
It is an up-to-date version of [Stable-Baselines](https://stable-baselines.readthedocs.io/en/master/), 
which is based on [OpenAI Baselines](https://github.com/openai/baselines).

###Tutorials:

[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) is a collection of pre-trained Reinforcement Learning 
agents using Stable-Baselines3. It also provides basic scripts for training, evaluating agents, tuning hyperparameters 
and recording videos.\
Deep reinforcement learning is very sensitive to the choice of hyperparamters. Automatic hyperparameter optimization 
can be done using [Optuna](https://github.com/pfnet/optuna) or [PFRL](https://github.com/pfnet/pfrl).\
Consider using [*Weights and Biases*](https://wandb.ai/site) to track your experiments.\
See this [link](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html) for a list of tutorials implemented in Google Colab.

####[Training, saving, and loading](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb):

- First we need to define the model: `model = <ALG>(<policy_model>, <env>, *kwargs)`\
For example: `model = DQN('MlpPolicy', 'LunarLander-v2', verbose=1, exploration_final_eps=0.1, target_update_interval=250)`
- We use `MlpPolicy` because the input is a feature vector, not images.
- The type of action (discrete/continuous) is automatically deduced from the env action space.
- The `learn` function trains the agent: `model.learn(total_timesteps=<num>)`
- The `save` function saves the agent: `model.save(<name>)`
- The `load` function loads the agent: `model.load(<name>)`
- `evaluate_policy` is a helper function to evaluate a policy: `mean_reward, std_reward = evaluate_policy(<model>, <env>, *kwargs)`\
For example:
    ```
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    ```

####Wrappers:
[Wrappers](https://github.com/openai/gym/tree/master/gym/wrappers) are used to transform an environment in a modular way.
```
env = gym.make('Pong-v0')
env = MyWrapper(env)
```
They have many practical uses:
- Limit the length of the episodes.
- Normalize actions.

Quick tips for writing your own wrappers:
- Don't forget to call `super(class_name, self).__init__(env)` if you override the wrapper's `__init__` function
- You can access the inner environment with `self.unwrapped`
- You can access the previous layer using `self.env`
- The variables `metadata`, `action_space`, `observation_space`, `reward_range`, and `spec` are copied to `self` from the previous layer
- Create a wrapped function for at least one of the following: `__init__(self, env)`, `step`, `reset`, `render`, `close`, or `seed`
- Your layered function should take its input from the previous layer (`self.env`) and/or the inner layer (`self.unwrapped`)

####[Multiprocessing](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb):

- Multiprocessing can speed-up the training process by running several processes in parallel.
- To multiprocess RL training, we have to wrap the environment into a `SubprocVecEnv` object, 
which will take care of synchronising the processes. Each process will run an independent instance of the environment.
- The number of parallel processes is defined with the `num_cpu` variable.
- The `make_vec_env()` function creates a vectorized environment. It will instantiate the environments 
and make sure they are different (using different random seeds)\
For example:
    ```
    vec_env = make_vec_env(env_id, n_envs=num_cpu)
    
    model = A2C('MlpPolicy', vec_env, verbose=0)
    ```
- Then the model can be trained just like a single process model, but it will be faster.

####Callbacks:

A [callback](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html) 
is a set of functions that will be called at given stages of the training procedure. 
You can use callbacks to access internal state of the RL model during training. 
It allows one to do monitoring, auto saving, model manipulation, progress bars, etc.\
Some of the most useful built-in callbacks are:
- CheckpointCallback: periodically save the model. `callback=checkpoint_callback`
- EvalCallback: periodically evaluate the model and save the best one. `callback=eval_callback`

Other utilities:
- CallbackList: chain several callbacks (they will be called sequentially).
- StopTrainingOnRewardThreshold: Stop the training early based on reward threshold.
- EveryNTimesteps: Trigger a callback every *n* timesteps.

####[Monitor training](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb):

- The `Monitor` class is a wrapper for Gym environments. It is used to know the episode reward, length, time and other data.\
For example:
    ```
    from stable_baselines3.common.monitor import Monitor
    
    # Create log dir
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create and wrap the environment
    env = gym.make('LunarLanderContinuous-v2')
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)
    ```
- There are built-in plotting helpers in `stable_baselines3.common.results_plotter`

####[Normalization](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb):
- Normalizing input features may be essential to successful training of an RL agent 
(by default, images are scaled but not other types of input), for instance when training on PyBullet environments. 
For that, the wrapper [`VecNormalize`](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#stable_baselines3.common.vec_env.VecNormalize) 
exists and will compute a running average and standard deviation of input features (it can do the same for rewards).
    ```
    env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
    
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    ```
- Don't forget to save the `VecNormalize` statistics when saving then agent:
    ```
    log_dir = "/tmp/"
    model.save(log_dir + "ppo_halfcheetah")
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)
    
    # Load the agent
    model = PPO.load(log_dir + "ppo_halfcheetah")
    
    # Load the saved statistics
    env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
    env = VecNormalize.load(stats_path, env)
    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False
    ```

####[Advanced saving and loading](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/advanced_saving_loading.ipynb):
- When defining the model, you can automatically create an environment for evaluation by using the `create_eval_env` argument:
    ```
    model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1, learning_rate=1e-3, create_eval_env=True)
    ```
- Then, when you train the agent you can evaluate it periodically on the test env:
    ```
    # Evaluate the model every 1000 steps on 5 test episodes and save the evaluation to the logs folder
    model.learn(6000, eval_freq=1000, n_eval_episodes=5, eval_log_path="./logs/")
    ```
- You can save the policy independently from the model if needed. But note that if you don't save the complete model 
then you can't continue training afterward.
    ```
    policy = model.policy
    policy.save("sac_policy_pendulum.pkl")
    ```

####Recording:
Recording is very useful when training or evaluating on Google Colab. 
Use the [VecVideoRecorder](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#vecvideorecorder) 
wrapper to record a video of the episode. See the [getting_started notebook](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb) 
for more information.