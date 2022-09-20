# Weights and biases (W&B) Information

W&B is an experiment tracking tool for machine learning.
Its purpose is to help with reproducibility, analysis & visualization, and reliability.
While you train your model, W&B collects information in the background.

## SB3 integration:
wandb callback that can be used with SB3 code.
- Initialize a wandb run, create a SB3 model, add the wandb callback. wand automatically uploads the hyperparameters and results to your dashboard on the cloud.

## Experiment tracking:
- Initialize a new run at the top of your script with `wanb.init()`.
- Use `wand.config()` to save a dictionary of hyperparameters (to capture?) such as learning rate or model type.
- Use `wand.log()` to log metrics over time in a training loop, such as accuracy and loss.
- Use `wandb.log_artifact` to save the outputs of a run, like the model weights or a table of predections.
- Some things are logged automatically, including system metrics (CPU and GPU utilization, etc) and the command line inputs and outputs.

[`wandb.init(`](https://docs.wandb.ai/ref/python/init)
- `project`: name of the project where you're sending the new run (optional, default is "Uncategorized")
- `entity`: a username or team name where you are sending runs (optional, default is your username)
- [`config`](https://docs.wandb.ai/guides/track/config): a dictionary-like object for saving inputs to your job, like hyperparameters or other settings (optional).
      The config will show up on a table that you can use to group or filter runs.
- `save_code`: boolean to save the main script or notebook to W&B (optional, default is off)
- `group`: specify a group to organize individual runs into a larger experiment (optional)
- `job_type`: used to group runs further, e.g. into train or eval. (optional)
- `tags`: list of tags to filter runs (optional)
- `name`: name for the run (optional, default is a random two-word name)
- `notes`: a longer description of the run to remember what you were doing (optional)
- `dir`: an absolute path to a directory where the metadata will be stored (optional, default is ./wandb)
- `resume`: sets the resuming behavior when the run has the same ID as a previous run (default is None: overwrites previous data;
      True: if the previous run crashed it automatically resumes it; "allow": automatically resumes the run)
- `reinit`: allow multiple init() call in the same process (optional, default is False)
- `magic`:
- `config_exclude_keys`:
- `config_include_keys`:
- `anonymous`:
- `mode`: can be "online", "offline", or "disabled" (optional, default is online).
- `allow_val_change`: whether or not to allow config values to change after setting the keys once. 
            By default, an exception is thrown if a config value is overwritten.
            If you want to track something like a varying learning rate, use wandb.log() instead.
            (optional, default is False in scripts, True in Jupyter)
- `force`: 
- `sync_tensorboard`: synchronize wandb logs from tensorboard and save teh relevant events file (optional, default is False)
- `monitor_gym`: automatically log videos of environment when using OpenAI Gym (optional, default is False)
- `id`: a unique ID for this run, used for resuming. (optional string)

`)`

[`wandb.log(`](https://docs.wandb.ai/ref/python/log)
`"Log data from runs, such as scalars, images, video, histograms, plots, tables."`
- `data`: a dict of serializable python objects,  e.g. str, int, float, Tensor, dict, or any of the [wandb data types](https://docs.wandb.ai/ref/python/data-types). (optional)
- `commit`: boolean to save the metrics dict to the wandb server and increment the step. (optional, default is True, if set to False then
the metrics get updated, but won't be saved until log is called with commit=True).
- `step`: the global step in processing (optional int)
- `sync`: deprecated (does not affect behavior)

`)`

Example:
```python
import wandb

wandb.init()
wandb.log({"accuracy": 0.9, "epoch": 5})
```

#### Other:
[`wandb.watch(`](https://docs.wandb.ai/ref/python/watch)
`"Hooks into the torch model to collect gradients and topology"`
- `models`: the model to hook (can be a tuple of models)
- `criterion`: An optional loss value being optimized (default is None)
- `log`: "gradients", "parameters", "all", or None
- `log_freq`: log gradients and parameters every N batches
- `idx`: an index to be called when calling wand.watch on multiple models
- `log_graph`: log graph topology (bool)

`) -> wandb.Graph`

`wandb.save(`
`"Save files on wandb"`
- `glob_string`: relative or absolute path
- `base_path`:
- `policy`: "live" to upload the file as it changes, "now" to upload the file once now, "end" to only upload the file when the run ends.

## Hyperparameter tuning:
- config: define the variables and ranges to sweep over, pick a search strategy (grid, random, bayesian, plus techniques such as early stopping), 
	    pick the optimization metric (make sure you are logging this metric; also note that if you log this metric more than once per run, it only uses the last recorded value for each run), etc.
- initialize sweep: wandb.sweep(sweep_config)
- run the sweep agent: wandb.agent(sweep_id, function=train)

You can then see parameter importance plots and paralled coordinates plots (look for patterns).

[`wandb.sweep(`](https://docs.wandb.ai/ref/python/sweep)
`"Initializes a hyperparameter sweep (does not run it, only declares it)"`
- [`sweep`](https://docs.wandb.ai/guides/sweeps/configuration): configuration dict
- `entity`: username or team name where you are sending the runs to (optional).
- `project`: name of the project where you are sending the runs to (optional).

`) -> sweepID`

[`wandb.agent(`](https://docs.wandb.ai/ref/python/agent)
`"Runs the sweep"`
- `sweep_id`: sweep ID generated by wandb.sweep()
- `function`: a function to call instead of the "program" specified in the config (optional)
- `entity`:
- `project`:
- `count`: number of trials to run (optional)

`)`

Example:
```python
import wandb
sweep_configuration = {
    "name": "my-awesome-sweep",
    "metric": {"name": "accuracy", "goal": "maximize"},  # goal can be 'maximize' or 'minimize'
    "method": "grid",
    "parameters": {
        "a": {
            "values": [1, 2, 3, 4]
        }
    }
}

def my_train_func():
    # read the current value of parameter "a" from wandb.config
    wandb.init()
    a = wandb.config.a

    wandb.log({"a": a, "accuracy": a + 1})

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my_project")

# run the sweep
wandb.agent(sweep_id, function=my_train_func)
```

### Strategies for sweep:
- Start-off broad. Only narrow it down once you have a better idea of how the different values perform.
- First use random search to narrow-down your hyperparameter space (remove ranges that are just not good), 
then you can use grid search to fine-tune it.
- When using random search, for parameters such as `batch_size` and `learning_rate`, use `log_uniform` distributions instead of uniform distributions (`log_uniform` will sample orders of magnitude with equal probability).

## Data visualization:
Use *W&B Tables* to log, query, and analyze tabular data.
The fastest way to try Tables is to log a dataframe and see the Table UI: `wandb.log({"table": my_dataframe})`

```python
import wandb
import random
import math
# Set up data to log in custom charts
data = []
for i in range(100):
    data.append([i, random.random() + math.log(1 + i) + random.random()])

# Create a table with the columns to plot
table = wandb.Table(data=data, columns=["step", "height"])

# Use the table to populate various custom charts
line_plot = wandb.plot.line(table, x='step', y='height', title='Line Plot')
histogram = wandb.plot.histogram(table, value='height', title='Histogram')
scatter = wandb.plot.scatter(table, x='step', y='height', title='Scatter Plot')

# Log custom tables, which will show up in customizable charts in the UI
wandb.log({'line_1': line_plot, 
         'histogram_1': histogram, 
         'scatter_1': scatter})
```