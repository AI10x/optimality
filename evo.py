import torch
from torch.distributions import Distribution, MultivariateNormal

import evotorch
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver
from evotorch.logging import StdOutLogger

class PartiallyObservableDistribution(Distribution):
    """
    A wrapper around a torch.distributions.Distribution that allows for conditioning on partial observations.
    """
    def __init__(self, dist):
        if not isinstance(dist, MultivariateNormal):
            raise NotImplementedError("Currently only MultivariateNormal is supported.")
        self.dist = dist
        super().__init__(dist.batch_shape, dist.event_shape)

    @property
    def mean(self):
        return self.dist.mean

    @property
    def variance(self):
        return self.dist.variance

    def sample(self, sample_shape=torch.Size()):
        return self.dist.sample(sample_shape)

    def log_prob(self, value):
        """
        Computes the log probability of the observed values, marginalizing out the unobserved ones.
        """
        if not torch.is_tensor(value):
            raise ValueError("Value must be a torch.Tensor")

        # For a multivariate normal, the marginal distribution of a subset of
        # variables is also a normal distribution.
        observed_mask = ~torch.isnan(value)
        
        # We can compute the log_prob for each event separately.
        log_probs = torch.zeros(value.shape[0], device=value.device)
        for i in range(value.shape[0]):
            obs_mask_i = observed_mask[i]
            if not obs_mask_i.any():
                # If no observations, log_prob is 0 (log of probability 1 of an empty event)
                # or should this be the log of the total volume? Let's stick to 0 for now.
                continue

            # Marginalize out the unobserved dimensions
            marginal_dist = self._get_marginal(obs_mask_i)
            log_probs[i] = marginal_dist.log_prob(value[i, obs_mask_i])
            
        return log_probs

    def condition(self, value):
        """
        Conditions the distribution on the observed values.
        Returns a new distribution over the unobserved variables.
        """
        if not torch.is_tensor(value):
            raise ValueError("Value must be a torch.Tensor")

        observed_mask = ~torch.isnan(value)
        unobserved_mask = torch.isnan(value)

        if not unobserved_mask.any():
            raise ValueError("No unobserved values to condition on.")
        if not observed_mask.any():
            return self # Nothing to condition on

        # For now, let's handle single events (not batches)
        if value.dim() > 1 and value.shape[0] > 1:
            raise NotImplementedError("Conditioning on batches is not yet supported.")
        
        observed_mask = observed_mask.squeeze()
        unobserved_mask = unobserved_mask.squeeze()
        value = value.squeeze()

        # Extract parameters of the original distribution
        mu = self.dist.mean
        cov = self.dist.covariance_matrix

        # Select the observed and unobserved parts
        mu_obs = mu[observed_mask]
        mu_unobs = mu[unobserved_mask]
        
        cov_obs_obs = cov[observed_mask][:, observed_mask]
        cov_unobs_unobs = cov[unobserved_mask][:, unobserved_mask]
        cov_obs_unobs = cov[observed_mask][:, unobserved_mask]
        cov_unobs_obs = cov[unobserved_mask][:, observed_mask]

        # Compute the conditional distribution parameters
        # mu_cond = mu_unobs + cov_unobs_obs @ inv(cov_obs_obs) @ (value_obs - mu_obs)
        # cov_cond = cov_unobs_unobs - cov_unobs_obs @ inv(cov_obs_obs) @ cov_obs_unobs
        
        prec_obs_obs = torch.inverse(cov_obs_obs)
        
        mu_cond = mu_unobs + cov_unobs_obs @ prec_obs_obs @ (value[observed_mask] - mu_obs)
        cov_cond = cov_unobs_unobs - cov_unobs_obs @ prec_obs_obs @ cov_obs_unobs
        
        return MultivariateNormal(mu_cond, cov_cond)

    def _get_marginal(self, observed_mask):
        """
        Returns the marginal distribution for the observed dimensions.
        """
        mu = self.dist.mean
        cov = self.dist.covariance_matrix
        
        mu_marginal = mu[observed_mask]
        cov_marginal = cov[observed_mask][:, observed_mask]
        
        return MultivariateNormal(mu_marginal, cov_marginal)

if __name__ == '__main__':
    # Example Usage
    
    # 1. Define a base multivariate normal distribution
    mean = torch.tensor([0.0, 1.0, 2.0])
    covariance_matrix = torch.tensor([
        [1.0, 0.5, 0.2],
        [0.5, 1.2, 0.8],
        [0.2, 0.8, 1.5]
    ])
    base_dist = MultivariateNormal(mean, covariance_matrix)
    
    # 2. Wrap it in a PartiallyObservableDistribution
    po_dist = PartiallyObservableDistribution(base_dist)
    
    # 3. Create a partially observed sample
    # Let's say we observed the first and third dimensions
    observed_value = torch.tensor([[0.5, torch.nan, 2.5]])
    
    # 4. Calculate the log probability of the observations
    log_p = po_dist.log_prob(observed_value)
    print(f"Log probability of observed values: {log_p.item()}")

    # 5. Condition on the observed values to get a distribution over the unobserved ones
    conditional_dist = po_dist.condition(observed_value)
    
    print(f"Conditional distribution mean: {conditional_dist.mean}")
    print(f"Conditional distribution variance: {conditional_dist.variance}")
    
    # 6. Sample from the conditional distribution
    unobserved_sample = conditional_dist.sample()
    print(f"Sample for the unobserved variable: {unobserved_sample}")

    # 7. Use evotorch to find the most likely value of the unobserved variable using GeneticAlgorithm
    print("\n--- Using evotorch to find the most likely value with GeneticAlgorithm ---")

    def batch_f(x: torch.Tensor) -> torch.Tensor: #fit objective function with loss func
        """
        Batch evaluation function for evotorch.
        Takes a batch of candidate solutions and returns their log-probabilities.
        `x` is a tensor of shape (popsize, solution_length).
        """
        popsize = x.shape[0]

        # Create a batch of full tensors by repeating the observed value
        # and filling in the unobserved values from the solution batch `x`.
        full_values = observed_value.repeat(popsize, 1)
        unobserved_mask = torch.isnan(observed_value.squeeze())
        full_values[:, unobserved_mask] = x
        return base_dist.log_prob(full_values)

    problem = evotorch.Problem(
        objective_sense="max",
        objective_func=batch_f,
        solution_length=1,  # We are looking for one unobserved value
        initial_bounds=(-10.0, 10.0),
        device="cpu",
    )

    # Initialize the search algorithm
    searcher = GeneticAlgorithm(
                problem,
                popsize=100,
                operators=[
                    SimulatedBinaryCrossOver(problem, eta=20.0, tournament_size=5),
                    GaussianMutation(problem, stdev=0.01),
                ]
            )

    # Run the optimization
    logger = StdOutLogger(searcher)
    searcher.run(50)

    # Get the best solution
    best_solution = searcher.status["best"]
    print(f"Most likely value found by evotorch: {best_solution.values.item()}")
    print(f"Log probability at most likely value: {best_solution.evals.item()}")
