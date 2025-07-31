# Lab-Cognitive-Modelling-of-Biological-Agents

This project implements a modular maze environment for reinforcement learning, supporting various agent designs and a unified interface for training and evaluation.

## Project Structure
- `maze/`: Environment logic and rendering
- `agents/`: Different agent policies
- `training/`: Scripts to train and test agents

## Getting Started
(optional, not necessary) To install the environment in a custom directory, add a prefix with the desired file path to the end of the .yml file.

```yml
prefix: C:\Users\user\anaconda3
```

Create the environment using the following command:

```bash
conda env create -f "PATH_TO_lab_env.yml"
```

---

## Command-Line Usage

You can start the application using the command line with various modes and arguments.  
Run the main script with:

```bash
python main.py --mode <mode> [--agent <agent>] [--experiment_name <name>] [--maze_id <id>] [--port <port>]
```

### Modes

- `--mode optimize`  
  Run hyperparameter optimization for a specified agent.  
  Requires `--agent`.

- `--mode load`  
  Load and visualize a saved experiment.  
  Requires `--experiment_name`.

- `--mode multi`  
  Launch multiple experiment dashboards concurrently (for demo purposes).

- `--mode render`  
  Render a maze by its ID.  
  Requires `--maze_id`.

### Agent Types (for `--agent`)
- `bayesian`
- `noisy_perceptual`
- `noisy_neural`
- `noisy_both`
- `curious`
- `sr_dyna`
- `q_learning`

### Examples

**Optimize a Bayesian Q-Learning agent:**
```bash
python main.py --mode optimize --agent bayesian --port 8050
```

**Load a saved experiment:**
```bash
python main.py --mode load --experiment_name Bayesian_Q-learning_agent_optuna_20250715_223504_sampling_policy_instance --port 8050
```

**Render a specific maze:**
```bash
python main.py --mode render --maze_id 26
```

**Run multiple dashboards:**
```bash
python main.py --mode multi
```

---