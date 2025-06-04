from training.hyperparameter import Hyperparameter
from enums.agent_type import AgentType
from enums.hyperparam_opt_type import HyperparamOptType
from enums.score_metric import ScoreMetric
from enums.noise_mode import NoiseMode
from training.train_script import ExperimentResult
from training.experiment import Experiment
import optuna


class HyperparameterScheduler:
    def __init__(self,
                optimization_type:HyperparamOptType = HyperparamOptType.OPTUNA,
                agent_type: AgentType = AgentType.BAYESIAN_Q_LEARNING,
                noise_mode: NoiseMode = NoiseMode.NONE,
                opt_score_metric:ScoreMetric = ScoreMetric.MAX_REWARD,
                n_trials: int = 100,
                random_seed: int = 42
                ):
        """
        Initialize the hyperparameter scheduler.
        Args:
            optimization_type: Type of optimization to use (e.g., "grid_search", "random_search").
            agent_type: Type of agent for which to optimize hyperparameters.
            hyperparameter_ranges: Dictionary defining the ranges for each hyperparameter.
        """
        self.optimization_type = optimization_type
        self.agent_type = agent_type
        if noise_mode != NoiseMode.NONE and agent_type != AgentType.NOISY_AGENT:
            raise ValueError("Noise mode is not applicable for non-noisy agents.")
        
        self.noise_mode = noise_mode
        self.opt_score_metric = opt_score_metric
        self.n_trials = n_trials
        self.seed = random_seed  # Seed for reproducibility

    def _setup_hyperparam_ranges(self):
        if self.agent_type == AgentType.Q_LEARNING:
            self.hyperparameter_ranges: dict = {
                "alpha": (0.01, 0.1),
                "gamma": (0.9, 0.99),
                "epsilon": (0.1, 0.5)
            }

        elif self.agent_type == AgentType.BAYESIAN_Q_LEARNING:
            self.hyperparameter_ranges: dict = {
                    "alpha": (0.01, 0.1),
                    "gamma": (0.9, 0.99),
                    "epsilon": (0.1, 0.5),
                    "mu_init": (0.0, 1.0),
                    "sigma_sq_init": (0.1, 2.0),
                    "obs_noise_variance": (0.01, 0.5)
                }

        elif self.agent_type == AgentType.CURIOUS_AGENT:
            self.hyperparameter_ranges: dict = {
                "gamma": (0.9, 0.99),
                "epsilon": (0.1, 0.5),
                "mu_init": (0.0, 1.0),
                "sigma_sq_init": (0.1, 2.0),
                "obs_noise_variance": (0.01, 0.5),
                "curiosity_init": (0.1, 1.0),
                "alpha_C": (0.01, 0.1),
                "surprise_weight": (1e-5, 1.0),
                "novelty_weight": (1e-5, 1.0),
                "usefulness_weight": (1e-5, 1.0),
                "uncertainty_weight": (1e-5, 1.0),
                "beta_T": (0.01, 0.1),
                "beta_U": (0.01, 0.1),
                "beta_N": (0.01, 0.1)
            }

        elif self.agent_type == AgentType.NOISY_AGENT:
            if self.noise_mode == NoiseMode.NONE:
                raise ValueError("Noise mode must be specified for NoisyAgent.")
            self.hyperparameter_ranges: dict = {
                "gamma": (0.9, 0.99),
                "epsilon": (0.1, 0.5),
                "mu_init": (0.0, 1.0),
                "sigma_sq_init": (0.1, 2.0),
                "obs_noise_variance": (0.01, 0.5),
                "k_pn": (0.1, 1.0),
                "sigma_nn": (0.1, 1.0),
                "noise_mode": self.noise_mode
            }

        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization.
        Args:
            trial: A trial object from the optimization library.
        Returns:
            Score metric value based on the agent's performance.
        """
        # Extract hyperparameters from the trial
        if self.agent_type == AgentType.BAYESIAN_Q_LEARNING:
            hyperparameters = {
                "alpha": trial.suggest_float("alpha", *self.hyperparameter_ranges["alpha"]),
                "gamma": trial.suggest_float("gamma", *self.hyperparameter_ranges["gamma"]),
                "epsilon": trial.suggest_float("epsilon", *self.hyperparameter_ranges["epsilon"]),
                "mu_init": trial.suggest_float("mu_init", *self.hyperparameter_ranges["mu_init"]),
                "sigma_sq_init": trial.suggest_float("sigma_sq_init", *self.hyperparameter_ranges["sigma_sq_init"]),
                "obs_noise_variance": trial.suggest_float("obs_noise_variance", *self.hyperparameter_ranges["obs_noise_variance"])
            }
        elif self.agent_type == AgentType.Q_LEARNING:
            hyperparameters = {
                "alpha": trial.suggest_float("alpha", *self.hyperparameter_ranges["alpha"]),
                "gamma": trial.suggest_float("gamma", *self.hyperparameter_ranges["gamma"]),
                "epsilon": trial.suggest_float("epsilon", *self.hyperparameter_ranges["epsilon"])
            }
        elif self.agent_type == AgentType.CURIOUS_AGENT:
            hyperparameters = {
                "gamma": trial.suggest_float("gamma", *self.hyperparameter_ranges["gamma"]),
                "epsilon": trial.suggest_float("epsilon", *self.hyperparameter_ranges["epsilon"]),
                "mu_init": trial.suggest_float("mu_init", *self.hyperparameter_ranges["mu_init"]),
                "sigma_sq_init": trial.suggest_float("sigma_sq_init", *self.hyperparameter_ranges["sigma_sq_init"]),
                "obs_noise_variance": trial.suggest_float("obs_noise_variance", *self.hyperparameter_ranges["obs_noise_variance"]),
                "curiosity_init": trial.suggest_float("curiosity_init", *self.hyperparameter_ranges["curiosity_init"]),
                "alpha_C": trial.suggest_float("alpha_C", *self.hyperparameter_ranges["alpha_C"]),
                "surprise_weight": trial.suggest_float("surprise_weight", *self.hyperparameter_ranges["surprise_weight"]),
                "novelty_weight": trial.suggest_float("novelty_weight", *self.hyperparameter_ranges["novelty_weight"]),
                "uncertainty_weight": trial.suggest_float("uncertainty_weight", *self.hyperparameter_ranges["uncertainty_weight"]),
                "beta_T": trial.suggest_float("beta_T", *self.hyperparameter_ranges["beta_T"]),
                "beta_U": trial.suggest_float("beta_U", *self.hyperparameter_ranges["beta_U"]),
                "beta_N": trial.suggest_float("beta_N", *self.hyperparameter_ranges["beta_N"])
            }
            usefulness_weight = 1.0 - hyperparameters["surprise_weight"] - hyperparameters["novelty_weight"] - hyperparameters["uncertainty_weight"]
            if usefulness_weight < self.hyperparameter_ranges["usefulness_weight"][0] or usefulness_weight > self.hyperparameter_ranges["usefulness_weight"][1]:
                raise optuna.exceptions.TrialPruned("Sum of weights exceeded 1 or resulted in out-of-range usefulness_weight.")
            hyperparameters["usefulness_weight"] = usefulness_weight


        elif self.agent_type == AgentType.NOISY_AGENT:
            hyperparameters = {
                "gamma": trial.suggest_float("gamma", *self.hyperparameter_ranges["gamma"]),
                "epsilon": trial.suggest_float("epsilon", *self.hyperparameter_ranges["epsilon"]),
                "mu_init": trial.suggest_float("mu_init", *self.hyperparameter_ranges["mu_init"]),
                "sigma_sq_init": trial.suggest_float("sigma_sq_init", *self.hyperparameter_ranges["sigma_sq_init"]),
                "obs_noise_variance": trial.suggest_float("obs_noise_variance", *self.hyperparameter_ranges["obs_noise_variance"]),
                "k_pn": trial.suggest_float("k_pn", *self.hyperparameter_ranges["k_pn"]),
                "sigma_nn": trial.suggest_float("sigma_nn", *self.hyperparameter_ranges["sigma_nn"]),
                "noise_mode": self.noise_mode
            }

        # Run the experiment with the current hyperparameters
        experiment = Experiment(agent_type=self.agent_type, hyperparameters=Hyperparameter(**hyperparameters))
        result:ExperimentResult = experiment.run_experiment()

        # Return the score metric based on the optimization type
        return eval(f"result.{self.opt_score_metric.value}")
    
    def _run_optuna(self, n_trials: int = 100) -> tuple:
        """
        Executes hyperparameter optimization using Optuna.
        """
        print("\n" + "="*80)
        print(f"Starting Optuna Optimization for '{self.agent_type}' (maximizing '{self.opt_score_metric}')...")
        print("="*80)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", study_name=f"{self.agent_type}_optimization", sampler=sampler)
        study.optimize(self._objective, n_trials=n_trials)


        print("\n" + "="*80)
        print(f"Optuna Optimization Complete for '{self.agent_type}'.")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial value ({self.opt_score_metric}): {study.best_value:.6f}")
        print(f"Best hyperparameters: {study.best_params}")
        print("="*80 + "\n")
        return study.best_params, study.best_value
    

    def start_optimization(self) -> tuple:
        """
        Starts the hyperparameter optimization process based on the specified optimization type.
        """
        self._setup_hyperparam_ranges()

        if self.optimization_type == HyperparamOptType.OPTUNA:
            return self._run_optuna(n_trials=self.n_trials)
        elif self.optimization_type == HyperparamOptType.GRID_SEARCH:
            # Placeholder for Grid Search implementation
            raise NotImplementedError("Grid Search optimization is not yet implemented.")
        else:
            raise ValueError(f"Unsupported optimization type: {self.optimization_type}")
