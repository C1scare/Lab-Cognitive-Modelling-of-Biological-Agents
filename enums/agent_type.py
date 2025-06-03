from enum import Enum

class AgentType(Enum):
    """Enum representing different types of agents."""
    Q_LEARNING = "Q-learning agent"
    BAYESIAN_Q_LEARNING = "Bayesian Q-learning agent"
    NOISY_AGENT = "Noisy Bayesian Q-learning agent"
    CURIOUS_AGENT = "Curious Bayesian Q-learning agent"
