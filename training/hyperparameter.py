from pydantic import BaseModel, Field
from typing import Optional
from enums.noise_mode import NoiseMode

class Hyperparameter(BaseModel):
    """
    Represents a hyperparameter with a name, value, and type.
    """
    alpha: Optional[float] = Field(default=None, description="The learning rate for the agent.")
    gamma: Optional[float] = Field(default=None, description="The discount factor for future rewards.")
    epsilon: Optional[float] = Field(default=None, description="The exploration rate for ε-greedy policy.")
    mu_init: Optional[float] = Field(default=None, description="Initial mean for the Gaussian Q-value distributions.")
    sigma_sq_init: Optional[float] = Field(default=None, description="Initial variance for the Gaussian Q-value distributions.")
    obs_noise_variance: Optional[float] = Field(default=None, description="Assumed variance of the Bellman target 'y'.")
    k_pn: Optional[float] = Field(default=None, description="Parameter for the magnitude of perceptual noise.")
    sigma_nn: Optional[float] = Field(default=None, description="Variance of neural noise proportional to the magnitude of neural noise.")
    noise_mode: Optional[NoiseMode] = Field(default=None, description="Mode of noise to apply (e.g., 'PERCEPTUAL', 'NEURAL', 'BOTH').")

    curiosity_init: Optional[float] = Field(default=None, description="Initial curiosity value for the agent, used in curiosity-driven exploration.")
    alpha_C : Optional[float] = Field(default=None, description="Learning rate for curiosity.")
    surprise_weight: Optional[float] = Field(default=None, description="Weight for the surprise component in curiosity-driven exploration.")
    novelty_weight: Optional[float] = Field(default=None, description="Weight for the novelty component in curiosity-driven exploration.")
    usefulness_weight: Optional[float] = Field(default=None, description="Weight for the usefulness component in curiosity-driven exploration.")
    uncertainty_weight: Optional[float] = Field(default=None, description="Weight for the uncertainty component in curiosity-driven exploration.")
    beta_T: Optional[float] = Field(default=None, description="Uncertainty scaling factor for the curiosity-driven exploration.")
    beta_U: Optional[float] = Field(default=None, description="Usefulness scaling factor for the usefulness component in curiosity-driven exploration.")
    beta_N: Optional[float] = Field(default=None, description="Novelty scaling factor for the novelty component in curiosity-driven exploration.")

    episodes: Optional[int] = Field(default=500, description="Number of episodes for training the agent.")