"""
Takeoff World Model with Failure Prediction Environment

An extended safety-critical reinforcement learning environment that combines
world models, uncertainty quantification, and explicit failure prediction.
This environment provides rewards based on both epistemic uncertainty and
learned failure probabilities to encourage safe exploration while avoiding
predicted failure states.
"""

from os import path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.patches as patches
import torch
import math
import random


class Takeoff_WM_Failure(gym.Env):
    """
    World Model-based Takeoff Environment with Failure Prediction.
    
    This environment extends the base world model environment by incorporating
    explicit failure prediction capabilities. It uses a trained failure classifier
    to predict the probability of failure in different states and combines this
    with epistemic uncertainty to create a comprehensive safety reward signal.
    
    The reward function can operate in two modes:
    1. Failure-only: Uses only failure predictions for rewards
    2. Combined: Uses both uncertainty quantification and failure predictions
    
    Attributes:
        observation_space: Feature space from the world model (1536-dimensional)
        action_space: Joint action space (7-dimensional, bounded)
        failure_threshold: Threshold for failure classification (default: 0.7)
        config.use_uq: Flag to enable/disable uncertainty quantification in rewards
    """
    def __init__(self):
        """
        Initialize the Takeoff World Model environment with failure prediction.
        
        Sets up observation and action spaces, device configuration, and
        environment parameters for safe exploration using world models and
        explicit failure prediction.
        """
        self.render_mode = None
        self.time_step = 0.05  # Simulation time step in seconds
        
        # State bounds for position and orientation (currently unused but kept for compatibility)
        self.state_bounds_high = np.array([1., 1., np.pi])  # [x, y, theta]
        self.state_bounds_low = np.array([-1., -1., -np.pi])
        
        # GPU device for model inference
        self.device = 'cuda'

        # Observation space: World model features (latent state representation)
        # Shape: (1, 1, 1536) - batch_size, sequence_length, feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(1, 1, 1536), 
            dtype=np.float32
        )
        
        # Action space: 7-dimensional joint actions with symmetric bounds
        self.action_space = spaces.Box(
            low=-0.5, 
            high=0.5, 
            shape=(7,), 
            dtype=np.float32
        )
        
        self.image_size = 128  # Image size for rendering (if needed)
        self.failure_threshold = 0.7  # Threshold for binary failure classification
        
        # World model components (initialized later via set_wm)
        self.config = None
        self.wm = None
        self.dataset = None
        self.ensemble = None
        self.threshold = None
        self.feat_size = None
        self.feat = None  # Current feature representation
        self.latent = None  # Current latent state
        

    def set_wm(self, config: Any, wm: Any, ensemble: Any, dataset: Any) -> None:
        """
        Set up the world model components for the environment.
        
        Args:
            config: Configuration object containing model hyperparameters
            wm: Pre-trained world model for state transitions and predictions
            ensemble: Ensemble model for uncertainty quantification
            dataset: Training dataset for initialization and offline evaluation
        """
        self.config = config
        self.wm = wm
        self.dataset = dataset
        self.ensemble = ensemble
        self.threshold = self.config.ood_threshold
        
        # Calculate feature size based on dynamics configuration
        if self.config.dyn_discrete:
            # Discrete stochastic dynamics: combine discrete and deterministic components
            self.feat_size = (
                self.config.dyn_stoch * self.config.dyn_discrete + 
                self.config.dyn_deter
            )
        else:
            # Continuous stochastic dynamics: add stochastic and deterministic components
            self.feat_size = self.config.dyn_stoch + self.config.dyn_deter
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step using the world model and failure prediction.
        
        Args:
            action: Action to take in the environment (shape: [7])
            
        Returns:
            observation: Updated world model features 
            reward: Combined uncertainty and failure-based reward
            done: Episode termination flag (always False)
            truncated: Episode truncation flag (always False)
            info: Additional information dictionary (empty)
        """
        with torch.no_grad():
            # Ensure action is detached from computation graph
            action_tensor = action.detach()

            # Predict failure probability for current state
            failure_probabilities = self.failure_classifier(self.feat)[:, 0]  # Shape: [batch_size]
            
            # Transform failure probability to reward signal
            # Higher failure probability leads to lower reward
            failure_centered = 0.5 - failure_probabilities.clone()  # Center around 0
            failure_reward = 1.5 * torch.tanh(failure_centered)  # Apply nonlinear transformation

            # Determine final reward based on configuration
            if self.config.use_uq:
                # Combined mode: Use both uncertainty quantification and failure prediction
                
                # Calculate epistemic uncertainty for the current state-action pair
                epistemic_uncertainty = self.calculate_uncertainty(self.feat, action_tensor)
                uncertainty_reward = self.threshold - epistemic_uncertainty
                
                # Apply step function transformation to uncertainty rewards
                processed_uncertainty = uncertainty_reward.clone()
                processed_uncertainty[uncertainty_reward < -0.4] = -1.0  # High uncertainty penalty
                processed_uncertainty[uncertainty_reward > 0.4] = 1.0    # Low uncertainty bonus
                
                # Combine uncertainty and failure rewards (take minimum for safety)
                final_reward = self.calculate_reward(processed_uncertainty, failure_reward)
            else:
                # Failure-only mode: Use only failure prediction for rewards
                final_reward = failure_reward
            
            # Alternative: Use only uncertainty (commented out)
            # final_reward = processed_uncertainty
           
            # Update world model state by taking the action
            batched_action = action_tensor.reshape(
                (self.config.batch_size, self.config.batch_length, -1)
            )
            
            # Predict next latent state using world model dynamics
            self.latent = self.wm.dynamics.img_step(self.latent, batched_action)
            
            # Extract features from updated latent state
            self.feat = self.wm.dynamics.get_feat(self.latent)
            self.feat = self.feat.reshape(
                (self.config.batch_size * self.config.batch_length, -1)
            )

            # Environment continues indefinitely (no terminal states)
            done = False
            truncated = False
            info = {}

        return self.feat.cpu().numpy(), final_reward, done, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Samples a random trajectory from the dataset and uses it to initialize
        the world model's latent state and feature representation.
        
        Args:
            seed: Random seed for reproducibility (unused)
            options: Additional reset options (unused)
            
        Returns:
            observation: Initial world model features
            info: Reset information (empty)
        """
        super().reset(seed=seed)

        # Sample a random initial trajectory from the dataset
        initial_trajectory = next(self.dataset)

        with torch.no_grad():
            # Preprocess the trajectory data for the world model
            preprocessed_data = self.wm.preprocess(initial_trajectory)
            
            # Encode observations into latent representations
            encoded_observations = self.wm.encoder(preprocessed_data)
            
            # Use the world model dynamics to observe and create initial latent state
            self.latent, _ = self.wm.dynamics.observe(
                encoded_observations, 
                preprocessed_data["action"], 
                preprocessed_data["is_first"]
            )

            # Extract feature representation from the latent state
            self.feat = self.wm.dynamics.get_feat(self.latent)
            self.feat = self.feat.reshape(
                (self.config.batch_size * self.config.batch_length, -1)
            )

        return self.feat.cpu().numpy(), {}
    

      
    def calculate_uncertainty(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calculate epistemic uncertainty for a given state-action pair.
        
        Uses an ensemble model to estimate epistemic (model) uncertainty,
        which indicates how confident the model is about its predictions
        in the given state-action region.
        
        Args:
            features: Current world model features (latent state representation)
            action: Action to evaluate
            
        Returns:
            epistemic_uncertainty: Uncertainty estimates for each sample (on GPU)
        """
        with torch.no_grad():
            # Concatenate features and actions to create input for uncertainty estimation
            state_action_input = torch.concat([features, action], dim=-1)
            
            # Use ensemble model to estimate epistemic uncertainty
            uncertainty_estimates = self.ensemble.intrinsic_reward_penn(state_action_input)
            
            # Return uncertainty values (keep on GPU for further processing)
            return uncertainty_estimates[:, 0]
        
    
    def failure_classifier(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict failure probability for given features using the world model's failure head.
        
        Args:
            features: World model features (latent state representation)
            
        Returns:
            failure_probabilities: Failure probabilities in range [0, 1] for each sample
        """
        with torch.no_grad():
            # Use the pre-trained failure head from the world model
            failure_distribution = self.wm.heads["failure"](features)
            
            # Extract mean of the Bernoulli distribution (failure probability)
            failure_probabilities = failure_distribution.mean  # Shape: [batch_size, seq_len, 1]

            return failure_probabilities

    def calculate_reward(self, uncertainty_reward: torch.Tensor, failure_reward: torch.Tensor) -> torch.Tensor:
        """
        Combine uncertainty and failure rewards using a conservative (minimum) approach.
        
        This conservative approach ensures that the agent receives low rewards if either
        the uncertainty is high OR the failure probability is high, promoting safe
        exploration that avoids both uncertain and likely-to-fail regions.
        
        Args:
            uncertainty_reward: Reward based on epistemic uncertainty
            failure_reward: Reward based on failure prediction
            
        Returns:
            combined_reward: Final reward signal combining both safety signals
        """
        # Take element-wise minimum for conservative safety behavior
        combined_reward = torch.minimum(uncertainty_reward, failure_reward)
        
        return combined_reward