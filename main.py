from enums.agent_type import AgentType
from enums.noise_mode import NoiseMode
from training.hyperparameter import Hyperparameter
from training.experiment import Experiment
from training.hyperparameter_scheduler import HyperparameterScheduler
from maze.maze_scheduler import MazeScheduler
from enums.hyperparam_opt_type import HyperparamOptType
from enums.score_metric import ScoreMetric
import datetime
import pickle
import asyncio


def single_experiment_run(experiment: Experiment):
    """ Run a single experiment and create a dashboard for it."""
    experiment_result = experiment.run_experiment()
    app = experiment.create_dashboard(experiment_result)
    return app
    return None

def load_experiment(experiment_name: str, storage_path: str = "experiments/", port: int = 8050):
    """
    Load an experiment from a file.
    
    Args:
        experiment_name: Name of the experiment to load.
        storage_path: Path where the experiment results are stored.
    
    Returns:
        Experiment object loaded from the file.
    """
    file_path = f"{storage_path}{experiment_name}.pkl"
    with open(file_path, "rb") as f:
        experiment:Experiment = pickle.load(f)
    app = experiment.create_dashboard(experiment.scores)
    app.run(debug=False, port=port)


def optimize_run(hyperparameterScheduler: HyperparameterScheduler):
    """ Run the hyperparameter optimization process and execute the experiment with the best parameters."""
    best_params, best_value = hyperparameterScheduler.start_optimization()
    print(f"Best hyperparameters: {best_params}")
    print(f"Best value: {best_value}")
    
    # Run the experiment with the best hyperparameters
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparameters = Hyperparameter(**best_params)
    hyperparameters.noise_mode = hyperparameterScheduler.noise_mode
    hyperparameters.episodes = 150
    if hyperparameterScheduler.agent_type == AgentType.CURIOUS_AGENT:
        #hyperparameters.novelty_weight = 0.0
        #hyperparameters.usefulness_weight = 0.0
        #hyperparameters.uncertainty_weight = 0.0
        #hyperparameters.surprise_weight = 0.0
        #hyperparameters.alpha_C = 0.9
        pass
    experiment = Experiment(
        experiment_name=f"{hyperparameterScheduler.agent_type.value.replace(' ', '_')}_{hyperparameterScheduler.optimization_type.value}_{timestamp}_sampling_policy",
        agent_type=hyperparameterScheduler.agent_type,
        hyperparameters=hyperparameters,
        maze_scheduler=MazeScheduler(first=26, last=27, trials_maze=75),
        save_results=True
    )
    return single_experiment_run(experiment)

async def run_multiple_dashboards():
    """Run multiple dashboards concurrently."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, load_experiment, "Bayesian_Q-learning_agent_optuna_20250707_024648_split_run_instance", "experiments/", 8050),
        loop.run_in_executor(None, load_experiment, "Curious_Bayesian_Q-learning_agent_optuna_20250707_034600_split_run_instance", "experiments/", 8051),
        loop.run_in_executor(None, load_experiment, "Noisy_Bayesian_Q-learning_agent_optuna_20250707_025642_split_run_instance", "experiments/", 8052),
        loop.run_in_executor(None, load_experiment, "Noisy_Bayesian_Q-learning_agent_optuna_20250707_030838_split_run_instance", "experiments/", 8053),
        loop.run_in_executor(None, load_experiment, "Noisy_Bayesian_Q-learning_agent_optuna_20250707_032910_split_run_instance", "experiments/", 8054),
        loop.run_in_executor(None, load_experiment, "Q-learning_agent_optuna_20250707_040755_split_run_instance", "experiments/", 8055),
        loop.run_in_executor(None, load_experiment, "SR-Dyna_agent_optuna_20250707_040655_split_run_instance", "experiments/", 8056),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Initialize the experiment
    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.BAYESIAN_Q_LEARNING,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.AVERAGE_REWARD,
        n_trials=150
    )
    #app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.NOISY_AGENT,
        noise_mode=NoiseMode.PERCEPTUAL,
        opt_score_metric=ScoreMetric.AVERAGE_REWARD,
        n_trials=150
    )
    #app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.NOISY_AGENT,
        noise_mode=NoiseMode.NEURAL,
        opt_score_metric=ScoreMetric.AVERAGE_REWARD,
        n_trials=150
    )
    #app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.NOISY_AGENT,
        noise_mode=NoiseMode.BOTH,
        opt_score_metric=ScoreMetric.AVERAGE_REWARD,
        n_trials=150
    )
    #app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.CURIOUS_AGENT,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.AVERAGE_REWARD,
        n_trials=150
    )
    app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.SR_DYNA_AGENT,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.AVERAGE_REWARD,
        n_trials=150
    )
    #app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.Q_LEARNING,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.AVERAGE_REWARD,
        n_trials=150
    )
    #app = optimize_run(scheduler)
    app.run(debug=False, port=8050)
    
    #load_experiment("Curious_Bayesian_Q-learning_agent_optuna_20250710_121711_split_run_instance")

    #asyncio.run(run_multiple_dashboards())

    
