from enums.agent_type import AgentType
from enums.noise_mode import NoiseMode
from training.hyperparameter import Hyperparameter
from training.experiment import Experiment
from training.hyperparameter_scheduler import HyperparameterScheduler
from maze.maze_scheduler import MazeScheduler
from enums.hyperparam_opt_type import HyperparamOptType
from enums.score_metric import ScoreMetric
from maze.basic_maze import BasicMaze
import datetime
import pickle
import asyncio
import argparse


def single_experiment_run(experiment: Experiment):
    """
    Run a single experiment and create a dashboard for it.

    Args:
        experiment: An Experiment object to run.

    Returns:
        A Dash app instance for interactive visualization of the experiment results.
    """
    experiment_result = experiment.run_experiment()
    app = experiment.create_dashboard(experiment_result)
    return app

def load_experiment(experiment_name: str, storage_path: str = "results", port: int = 8050):
    """
    Load an experiment from a file and launch its dashboard.

    Args:
        experiment_name: Name of the experiment to load.
        storage_path: Path where the experiment results are stored.
        port: Port to run the dashboard on.

    Returns:
        None. Launches the Dash app for the loaded experiment.
    """
    print(f"Loading experiment: {experiment_name} from {storage_path}\n \
          This may take a few seconds depending on the size of the experiment data.")
    file_path = f"{storage_path}/{experiment_name}.pkl"
    with open(file_path, "rb") as f:
        experiment:Experiment = pickle.load(f)
    app = experiment.create_dashboard(experiment.scores)
    app.run(debug=False, port=port)


def optimize_run(hyperparameterScheduler: HyperparameterScheduler):
    """
    Run the hyperparameter optimization process and execute the experiment with the best parameters.

    Args:
        hyperparameterScheduler: HyperparameterScheduler instance to optimize and run.

    Returns:
        A Dash app instance for the experiment run with the best hyperparameters.
    """
    best_params, best_value = hyperparameterScheduler.start_optimization()
    print(f"Best hyperparameters: {best_params}")
    print(f"Best value: {best_value}")

    # Run the experiment with the best hyperparameters
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparameters = Hyperparameter(**best_params)
    hyperparameters.noise_mode = hyperparameterScheduler.noise_mode
    hyperparameters.episodes = 150

    # Optimized hyperparameters for Bayesian Q-learning agent, which can be used and set here for training
    #hyperparameters.gamma = 0.9614119708167211
    #hyperparameters.epsilon = 0.13044035930570644
    #hyperparameters.mu_init = 0.5864763204375096
    #hyperparameters.sigma_sq_init = 1.7040201847402174
    #hyperparameters.obs_noise_variance = 0.06525897001368647

    experiment = Experiment(
        experiment_name=f"{hyperparameterScheduler.agent_type.value.replace(' ', '_')}_{hyperparameterScheduler.optimization_type.value}_{timestamp}_sampling_policy",
        agent_type=hyperparameterScheduler.agent_type,
        hyperparameters=hyperparameters,
        maze_scheduler=MazeScheduler(first=26, last=27, trials_maze=75),
        save_results=True
    )
    return single_experiment_run(experiment)

async def run_multiple_dashboards():
    """
    Run multiple experiment dashboards concurrently on different ports.

    This function demonstrates how to launch several Dash apps in parallel for
    different experiment result files.

    Returns:
        None. Launches multiple Dash apps.
    """
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, load_experiment, "Bayesian_Q-learning_agent_optuna_20250715_224136_sampling_policy_instance", "results", 8050),
        loop.run_in_executor(None, load_experiment, "Curious_Bayesian_Q-learning_agent_optuna_20250716_014733_sampling_policy_instance", "results", 8051),
        loop.run_in_executor(None, load_experiment, "Perceptual_Noisy_Bayesian_Q-learning_agent_optuna_20250715_232559_sampling_policy_instance_kPN01", "results", 8052),
        loop.run_in_executor(None, load_experiment, "Neural_Noisy_Bayesian_Q-learning_agent_optuna_20250715_235845_sampling_policy_instance_sNN02", "results", 8053),
        loop.run_in_executor(None, load_experiment, "Both_Noisy_Bayesian_Q-learning_agent_optuna_20250716_025934_sampling_policy_instance_bhyp", "results", 8054),
        loop.run_in_executor(None, load_experiment, "Q-learning_agent_optuna%150_250%20026%5027_maxsteps%30_instance", "results", 8055),
        loop.run_in_executor(None, load_experiment, "SR-Dyna_agent_optuna%150_250%20026%5027_maxsteps%30_instance", "results", 8056),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    """
    Command-line entry point for running, optimizing, loading, or rendering experiments.

    Use --mode to select operation:
      - optimize: Run hyperparameter optimization for a specified agent.
      - load: Load and visualize a saved experiment.
      - multi: Launch multiple experiment dashboards concurrently.
      - render: Render a maze by its ID.
    """
    parser = argparse.ArgumentParser(description="Run or optimize experiments.")
    parser.add_argument("--mode", choices=["optimize", "load", "multi", "render"], required=True, help="Operation mode")
    parser.add_argument("--agent", choices=[
        "bayesian", "noisy_perceptual", "noisy_neural", "noisy_both", "curious", "sr_dyna", "q_learning"
    ], help="Agent type for optimization")
    parser.add_argument("--experiment_name", type=str, help="Experiment name to load")
    parser.add_argument("--maze_id", type=int, help="Maze id to render")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash app")
    args = parser.parse_args()

    agent_map = {
        "bayesian":      (AgentType.BAYESIAN_Q_LEARNING, NoiseMode.NONE),
        "noisy_perceptual": (AgentType.NOISY_AGENT, NoiseMode.PERCEPTUAL),
        "noisy_neural":     (AgentType.NOISY_AGENT, NoiseMode.NEURAL),
        "noisy_both":       (AgentType.NOISY_AGENT, NoiseMode.BOTH),
        "curious":      (AgentType.CURIOUS_AGENT, NoiseMode.NONE),
        "sr_dyna":      (AgentType.SR_DYNA_AGENT, NoiseMode.NONE),
        "q_learning":   (AgentType.Q_LEARNING, NoiseMode.NONE),
    }

    mode: str = getattr(args, "mode")
    agent: str | None = getattr(args, "agent", None)
    experiment_name: str | None = getattr(args, "experiment_name", None)
    maze_id: int | None = getattr(args, "maze_id", None)
    port: int = getattr(args, "port", 8050)

    if mode == "optimize":
        if not agent:
            print("Please specify --agent for optimize mode.")
        else:
            agent_type, noise_mode = agent_map[agent]
            scheduler = HyperparameterScheduler(
                optimization_type=HyperparamOptType.OPTUNA,
                agent_type=agent_type,
                noise_mode=noise_mode,
                opt_score_metric=ScoreMetric.AVERAGE_REWARD,
                n_trials=150
            )
            app = optimize_run(scheduler)
            app.run(debug=False, port=port)
    elif mode == "load":
        if not experiment_name:
            print("Please specify --experiment_name for load mode.")
        else:
            # If the experiment name contains a folder, split it
            if "/" in experiment_name or "\\" in experiment_name:
                # Handle both forward and backward slashes for cross-platform compatibility
                folder, experiment_name = experiment_name.replace("\\", "/").split("/")
            else:
                folder, experiment_name = experiment_name.split("/") if "/" in experiment_name else (experiment_name, "results")
            load_experiment(experiment_name, port=port, storage_path=folder)
    elif mode == "multi":
        asyncio.run(run_multiple_dashboards())
    elif mode == "render":
        if maze_id is None:
            print("Please specify --maze_id for render mode.")
        else:
            BasicMaze.render_maze_figure(maze_id)
