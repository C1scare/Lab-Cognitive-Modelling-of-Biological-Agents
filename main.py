from enums.agent_type import AgentType
from enums.noise_mode import NoiseMode
from training.hyperparameter import Hyperparameter
from training.experiment import Experiment
from training.hyperparameter_scheduler import HyperparameterScheduler
from enums.hyperparam_opt_type import HyperparamOptType
from enums.score_metric import ScoreMetric
import datetime
import pickle


def single_experiment_run(experiment: Experiment):
    experiment_result = experiment.run_experiment()
    #app = experiment.create_dashboard(experiment_result)
    #return app
    return None

def load_experiment(experiment_name: str, storage_path: str = "experiments/"):
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
    app.run(debug=False, port=8050)


def optimize_run(hyperparameterScheduler: HyperparameterScheduler):
    best_params, best_value = hyperparameterScheduler.start_optimization()
    print(f"Best hyperparameters: {best_params}")
    print(f"Best value: {best_value}")
    
    # Run the experiment with the best hyperparameters
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparameters = Hyperparameter(**best_params)
    hyperparameters.noise_mode = hyperparameterScheduler.noise_mode
    if hyperparameterScheduler.agent_type == AgentType.CURIOUS_AGENT:
        hyperparameters.usefulness_weight = 1.0 - hyperparameters.surprise_weight - hyperparameters.novelty_weight - hyperparameters.uncertainty_weight
    experiment = Experiment(
        experiment_name=f"{hyperparameterScheduler.agent_type.value.replace(' ', '_')}_{hyperparameterScheduler.optimization_type.value}_{timestamp}",
        agent_type=hyperparameterScheduler.agent_type,
        hyperparameters=hyperparameters,
        save_results=True
    )
    return single_experiment_run(experiment)


if __name__ == "__main__":
    # Initialize the experiment
    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.BAYESIAN_Q_LEARNING,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.NOISY_AGENT,
        noise_mode=NoiseMode.PERCEPTUAL,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.NOISY_AGENT,
        noise_mode=NoiseMode.NEURAL,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.NOISY_AGENT,
        noise_mode=NoiseMode.BOTH,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.CURIOUS_AGENT,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.SR_DYNA_AGENT,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    app = optimize_run(scheduler)
    #app.run(debug=False, port=8050)
    
    #load_experiment("Q-learning_agent_optuna_20250625_234305_instance")

    
