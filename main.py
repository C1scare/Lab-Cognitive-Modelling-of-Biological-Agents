from experiment.experiment import Experiment
from experiment.hyperparameters import Hyperparameters
from agents.agent_factories import q_learning_factory

if __name__ == "__main__":
    hp = Hyperparameters.from_json("experiment/config/test.json")
    exp = Experiment(
        hyperparameters=hp,
        agent_factory=q_learning_factory,
        #agent_path="results/q_learning_factory/2025-06-04_00-23-55/agent_trained.pkl"
    )
    exp.run()
