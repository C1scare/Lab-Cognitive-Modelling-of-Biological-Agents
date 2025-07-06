from enums.agent_type import AgentType
from enums.noise_mode import NoiseMode
from training.hyperparameter import Hyperparameter
from training.experiment import Experiment
from training.hyperparameter_scheduler import HyperparameterScheduler
from enums.hyperparam_opt_type import HyperparamOptType
from enums.score_metric import ScoreMetric
import datetime
import pickle
import asyncio


def single_experiment_run(experiment: Experiment):
    experiment_result = experiment.run_experiment()
    #app = experiment.create_dashboard(experiment_result)
    #return app
    return None

async def load_experiment(experiment_name: str, storage_path: str = "experiments/", port:int=8050):
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

async def main():
    """
    The main coroutine that schedules and runs all tasks concurrently.
    """
    # A list of tuples, each containing the arguments for one call
    experiments_to_run = [
        ("Bayesian_Q-learning_agent_optuna_20250703_021348_instance", 8050),
        ("Curious_Bayesian_Q-learning_agent_optuna_20250703_023646_instance", 8051),
        ("Noisy_Bayesian_Q-learning_agent_optuna_20250703_021834_instance", 8052),
        ("Noisy_Bayesian_Q-learning_agent_optuna_20250703_022440_instance", 8053),
        ("Noisy_Bayesian_Q-learning_agent_optuna_20250703_022959_instance", 8054),
        ("Q-learning_agent_optuna_20250703_093138_instance", 8055),
        ("SR-Dyna_agent_optuna_20250703_024259_instance", 8056),
    ]

    # Create a list of tasks that can be run concurrently.
    # `asyncio.create_task` schedules the coroutine to run on the event loop.
    tasks = [
        asyncio.create_task(load_experiment(name, port)) 
        for name, port in experiments_to_run
    ]
    
    # `asyncio.gather` waits for all tasks in the list to complete.
    #results = await asyncio.gather(*tasks)

    #for res in results:
    #    print(f"- {res}")


if __name__ == "__main__":
    # Initialize the experiment
    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.BAYESIAN_Q_LEARNING,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    #app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.NOISY_AGENT,
        noise_mode=NoiseMode.PERCEPTUAL,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    #app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.NOISY_AGENT,
        noise_mode=NoiseMode.NEURAL,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    #app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.NOISY_AGENT,
        noise_mode=NoiseMode.BOTH,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    #app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.CURIOUS_AGENT,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    #app = optimize_run(scheduler)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.SR_DYNA_AGENT,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.SUCCESS_RATE,
        n_trials=150
    )
    #app = optimize_run(scheduler)
    #app.run(debug=False, port=8050)
    
    asyncio.run(main())

    
