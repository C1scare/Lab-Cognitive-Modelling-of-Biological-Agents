from enums.agent_type import AgentType
from enums.noise_mode import NoiseMode
from training.hyperparameter import Hyperparameter
from training.experiment import Experiment
from training.hyperparameter_scheduler import HyperparameterScheduler
from enums.hyperparam_opt_type import HyperparamOptType
from enums.score_metric import ScoreMetric


def single_experiment_run(experiment: Experiment):
    experiment_result = experiment.run_experiment()
    fig = experiment.create_dashboard(experiment_result)
    fig.show()


def optimize_run(hyperparameterScheduler: HyperparameterScheduler):
    best_params, best_value = hyperparameterScheduler.start_optimization()
    print(f"Best hyperparameters: {best_params}")
    print(f"Best value: {best_value}")
    
    # Run the experiment with the best hyperparameters
    experiment = Experiment(
        agent_type=hyperparameterScheduler.agent_type,
        hyperparameters=Hyperparameter(**best_params)
    )
    single_experiment_run(experiment)


if __name__ == "__main__":
    # Initialize the experiment
    experiment = Experiment(
        agent_type=AgentType.BAYESIAN_Q_LEARNING,
        hyperparameters=Hyperparameter(
            alpha=0.1,
            gamma=0.99,
            epsilon=0.2,
            mu_init=0.0,
            sigma_sq_init=2.0,
            obs_noise_variance=0.1
        ),
    )
    #single_experiment_run(experiment)

    scheduler = HyperparameterScheduler(
        optimization_type=HyperparamOptType.OPTUNA,
        agent_type=AgentType.CURIOUS_AGENT,
        noise_mode=NoiseMode.NONE,
        opt_score_metric=ScoreMetric.MAX_REWARD,
        n_trials=10
    )
    optimize_run(scheduler)

    
