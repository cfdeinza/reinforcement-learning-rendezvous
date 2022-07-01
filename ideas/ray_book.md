# Learning Ray (preprint)

Ray is a distributed computing framework for Python. You can efficiently parallelize Pyton programs on your laptop, 
and run the code you tested locally on a cluster practically without any changes. It has three main layers:
- the core engine
- the high-level libraries
- the ecosystem

Anyscale is the company behind Ray. Ray is used by companies such as Microsoft, Amazon, and Uber (see [user stories](https://www.anyscale.com/user-stories)).

In short, Ray sets up and manages clusters of computers so that you can run distributed tasks on them. A Ray cluster consists of nodes that are connected to each other via a network. 
You program against the so-called *driver*, the program root, which lives on the head node. The driver can run jobs (a collection of tasks), 
that are run on the nodes in the cluster. Specifically, the individual tasks of a job are run on worker processes on worker nodes.

A Ray cluster can also be a local cluster (i.e. a cluster cosisting of just your own computer). In this case, there is just one node (the head node), 
which has the driver process and some worker processes. The default number of worker processes is the number of CPUs available on your machine.

## Ray RLlib:
RLlib has a rich library of advanced RL algorithms to choose from. It works with both Pytorch and Tensorflow. 
It is distributed by default, so you can scale the training process to as many nodes as you want. 
All RLlib algorithms can be tuned with *Ray Tune*.

### Installation:
Ray requires Python 3.7 or later. To install Ray, run the following command: (it does not have to be version 1.9.0)
```commandline
pip install "ray[rllib]"==1.9.0
```
Once installed, you can easily import and initialize Ray as follows:
```python
import ray
ray.init()
```
This starts a Ray cluster on your local machine. It can utilize all the cores available on your computer as workers. 
If you want to run Ray on a real cluster, you would have to pass more arguments to `ray.init`, but the rest of your code would stay the same.

Our environment needs to have certain interfaces so that the agent can interact with it. 
RLlib uses the most known and widely-used library for RL environments: `gym`, an open-source Python project from OpenAI. 
Make sure you have it installed as well.

### Using the CLI:
First specify the training configuration in a `yaml` file such as `custom.yml`, as shown below:
```yaml
custom_env:
    env: custom_env.CustomEnv
    run: DQN
    checkpoint_freq: 1
    stop:
        timesteps_total: 10000
```
In the YAML file we specify the relative path to our environment class, the learning algorithm, the frequency at which we store checkpoints, and the stopping condition for the training. 
There are many other configuration parameters that can be specified in the file. 
After making the configuration file, you can start an RLlib training run with the following command:
```commandline
rllib train -f custom.yml
```
Ray will write the training results to a `logdir` directory located at `~/ray_results/custom_env`. 
Within that folder you will find another directory that starts with `DQN_custom_env.CustomEnv_` and contains an identifier for the experiment and the current date and time. 
Within that directory you will find other directories starting with a checkpoint prefix. 
To evaluate our trained algorithm, use:
```commandline
rllib evaluate ~/ray_results/custom_env/DQN_custom_env.CustomEnv_id_date/checkpoint_000010/checkpoint10 --run DQN --env custom_env.CustomEnv --steps 100
```
(Note that the algorithm and the environment need to match the ones used during training)

### Using the Python API:
To run workloads with RLlib from Python, your main entrypoint is the `Trainer` class. Specifically, you want to use the trainer corresponding to your algorithm (e.g. `DQNTrainer`). 
RLlib trainers are highly-configurable, but you can also initialize them without having to tweak any configuration parameters.

```python
from ray.tune.logger import pretty_print
from custom_env import Rendezvous3DOF
from ray.rllib.agents.dqn import DQNTrainer

trainer = DQNTrainer(env=Rendezvous3DOF, config={"num_workers": 4})

config = trainer.get_config()
print(pretty_print(config))

for i in range(10):
    result = trainer.train()

print(pretty_print(result))
```
In this case, we passed a `config` dictionary to the trainer constructor to tell it to use four workers in total. 
This means that the DQNTrainer will spawn four Ray actors, each using a CPU kernel, to train our DQN algorithm in parallel. 
Then we use a `for` loop to train the algorithm for ten iterations in total. 
Note that if you set the number of workers to zero, only the local worker on the head node will be created, and all training is done there. 
This can be useful for debugging, as no additional Ray actor processes are spawned. 

### Checkpointing:
Creating checkpoints of the model is very useful to ensure that you can recover your work in case of a crash, 
or simply to track training process peristently. You can create a checkpoint of your RLlib trainer at any point in the training process by calling `trainer.save()`. 
Once you have a checkpoint, you can easily restore your trainer with it, and evaluating a model is as simple as calling `trainer.evaluate(checkpoint)`.
```python
checkpoint = trainer.save()
print(checkpoint)

evaluation = trainer.evaluate(checkpoint)
print(pretty_print(evaluation))

restored_trainer = DQNTrainer(env=Rendezvous3DOF)
restored_trainer.restore(checkpoint)
```
You can also just call `trainer.evaluate()` without creating a checkpoint first, but it is usually a good practice to use checkpoints.

### Computing actions:
You can use the trainer to directly compute actions given the current state of the environment:
```python
env = Rendezvous3DOF()
done = False
total_reward = 0
observation = env.reset()

while not done:
    action = trainer.compute_single_action(observation) 
    observation, reward, done, info = env.step(action)
    total_reward += reward
```
To compute many actions at once, you can use the `compute_actions` method instead, which takes a dictionary of observations and produces a dictionary of actions:
```python
actions = trainer.compute_actions({'obs1': 1, 'obs2': 2})
print(actions)  # {'obs1': 10, 'obs2': 20}
```

### Accessing policies and models:
```python
policy = trainer.get_policy()
print(policy.get_weights())  # returns the parameters of the model

model = policy.model
```

If we used multiple workers, then there is not just one model, but a collection of four models that we trained on separate Ray workers. 
We can access all the models like this:
```python
workers = trainer.workers
workers.foreach_worker(lambda remote_trainer: remote_trainer.get_policy().get_weights())
```
You can use this to set model parameters as well, or otherwise configure workers. 
Every RLlib model obtained from a policy has a `base_model` that has a summary method to describe itself:
```python
model.base_model.summary()
```

### Configuring the experiments:
As we saw before, we can configure our training process using the `config` argument on our trainer. 

### Curriculum learning:
One of the most interesting features of RLlib is to provide a trainer with a *curriculum* to learn from. 
This means that instead of letting the trainer learn from arbitrary environment setups, we cherry pick states that are much easier to learn from, 
and then slowly introduce more difficult states. Building a learning curriculum in this manner is a great way to make your experiments converge on solutions quicker. 
The only thing you need to apply curriculum learning is a measure of which starting states are more difficult than others. 
For example, for the simple maze scenario, the distance of the seeker to the goal can be used as a measure of difficulty. 
An easy way to implement this measure is to make a `CurriculumEnv` environment that inherits from the `TaskSettableEnv` class. 
Then you need to define how to get the current difficulty (`get_task`) and how to set a required difficulty (`set_task`). 
We also need to define a function that tells the trainer when and how to set the task difficulty. For example, 
in the function below we simply increase the difficulty by one every 1000 time steps:
```python
def curriculum_fn(train_results, task_settable_env, env_ctx):
    time_steps = train_results.get('timesteps_total')
    difficulty = time_steps // 1000
    print(f'Current difficulty: {difficulty}')
    return difficulty
```
Finally, we need to add the curriculum function to our trainer config through the `env_task_fn` property.
```python
config = {
    'env': custom_env.CurriculumEnv,
    'env_task_fn': curriculum_fn,
}
```

### Other advanced topics:
- You can use *parametric action spaces* to "mask-out" undesired actions from the action space for each point in time.
- You can also have variable observation spaces.
