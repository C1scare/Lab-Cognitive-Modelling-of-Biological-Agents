from enum import Enum

class AgentType(Enum):
    """
    Enum for different types of agents used in experiments.
    """
    Q_LEARNING = "Q-learning agent"
    BAYESIAN_Q_LEARNING = "Bayesian Q-learning agent"
    NOISY_AGENT = "Noisy Bayesian Q-learning agent"
    CURIOUS_AGENT = "Curious Bayesian Q-learning agent"
    SR_DYNA_AGENT = "SR-Dyna agent"
