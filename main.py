from enums.agent_type import AgentType
from enums.noise_mode import NoiseMode
from training.hyperparameter import Hyperparameter
from training.experiment import Experiment
from training.hyperparameter_scheduler import HyperparameterScheduler
from enums.hyperparam_opt_type import HyperparamOptType
from enums.score_metric import ScoreMetric
import datetime
import asyncio


def single_experiment_run(experiment: Experiment):
    experiment_result = experiment.run_experiment()
    app = experiment.create_dashboard(experiment_result)
    return app


def optimize_run(hyperparameterScheduler: HyperparameterScheduler):
    best_params, best_value = hyperparameterScheduler.start_optimization()
    print(f"Best hyperparameters: {best_params}")
    print(f"Best value: {best_value}")
    
    # Run the experiment with the best hyperparameters
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparameters = Hyperparameter(**best_params)
    hyperparameters.noise_mode = hyperparameterScheduler.noise_mode
    experiment = Experiment(
        experiment_name=f"{hyperparameterScheduler.agent_type.value}_{timestamp}",
        agent_type=hyperparameterScheduler.agent_type,
        hyperparameters=hyperparameters,
        save_results=True
    )
    return single_experiment_run(experiment)


if __name__ == "__main__":
    # Initialize the experiment
    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.NOISY_AGENT,
        noise_mode=NoiseMode.NEURAL,
        opt_score_metric=ScoreMetric.MAX_REWARD,
        n_trials=10
    )
    app = optimize_run(scheduler)
    app.run(debug=False, port=8050)

    
