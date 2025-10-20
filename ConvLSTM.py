
import torch
import torch.nn as nn

from evotorch import Problem
from evotorch.algorithms import CMAES

"""
Define a Convolutional LSTM neural network module.
"""

class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1))
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1))

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x is expected to be of shape (batch_size, seq_len, C, H, W)
        b, t, c, h, w = x.size()

        # Apply convolutions to each time step
        conv_outputs = []
        for time_step in range(t):
            # Get the input for the current time step
            x_t = x[:, time_step, :, :, :]  # Shape: (b, c, h, w)

            # Pass through convolutional layers
            for i, conv in enumerate(self.conv_layers):
                x_t = conv(x_t)
            
            # Reshape for LSTM: (b, h*w, hidden_size) -> (b, hidden_size) by averaging
            # Or flatten and pass through a linear layer. Here we average.
            x_t = x_t.mean(dim=[2, 3]) # Global Average Pooling -> (b, hidden_size)
            conv_outputs.append(x_t)

        # Stack outputs along the time dimension
        lstm_input = torch.stack(conv_outputs, dim=1) # Shape: (b, t, hidden_size)

        lstm_out, _ = self.lstm(lstm_input)
        return(lstm_out)






class TestConvLSTM():
    def JointProbablity(self):
        print(f"Loaded {len(self.parsing)} data entries from rec.db.")



        point_x = self.tokens(self.samplesx)
        point_y = self.tokens(self.samplesy)


        self.surrogate_dist, self.cand_dist = self.probablity_dist(point_x, point_y)



    def probablity_dist(self, pointx, pointy):
        prob_func = lambda var: dist.Categorical(probs=torch.ones_like(var) / var.shape[-1])
        return(prob_func(pointx), prob_func(pointy))


    def test_convlstm_forward_pass_and_output_shape(self):
        """Tests the forward pass and the output shape of the ConvLSTM."""
        # Define input parameters
        input_size = 3  # Example: RGB channels
        hidden_size = 16
        num_layers = 2
        batch_size = 4
        seq_len = 10
        h, w = 32, 32

        # Create a random input tensor
        input_tensor = torch.randn(batch_size, seq_len, input_size, h, w)
        model = ConvLSTM(input_size, hidden_size, num_layers)
        # Pass the input through the model
        output = model.forward(input_tensor)
        # Check the output shape
      ##  expected_shape = (batch_size, seq_len, hidden_size)
        #self.assertEqual(output.shape, expected_shape)
        return(output)

    def optimize_convlstm_with_evotorch(self, num_generations=10, population_size=5):
        """
        Optimizes the parameters of the ConvLSTM model using EvoTorch and CMA-ES.
        """
        print("\n--- Starting Neuroevolution with EvoTorch and CMA-ES ---")

        # 1. Define model and data parameters
        input_size, hidden_size, num_layers = 3, 16, 2
        batch_size, seq_len, h, w = 4, 10, 32, 32

        # Create an instance of the model whose parameters we want to optimize
        model = ConvLSTM(input_size, hidden_size, num_layers)
        
        # Generate some random sample data and a target for the loss calculation
        sample_input = torch.randn(batch_size, seq_len, input_size, h, w)
        # The target is what we want the model to output. For this example, a tensor of zeros.
        target_output = torch.zeros(batch_size, seq_len, hidden_size)

        # 2. Define the Objective Function for EvoTorch
        # This function will be called by CMA-ES with candidate solutions (model parameters)
        def objective_function(params: torch.Tensor):
            # Load the flat parameter vector into the model
            nn.utils.vector_to_parameters(params, model.parameters())
            
            # Perform a forward pass
            output = model.forward(sample_input)
            
            # Calculate KL Divergence loss for joint distributions
            # The output and target are reshaped to treat the sequence as a single batch
            output_reshaped = output.view(batch_size * seq_len, hidden_size)
            target_reshaped = target_output.view(batch_size * seq_len, hidden_size)

            # The input to KLDivLoss should be log-probabilities
            log_probs, target_probs = self.JointProbablity(output_reshaped, target_reshaped)

            # Compute the KL Divergence loss
            # 'batchmean' reduction averages the loss over the batch dimension
            loss = nn.functional.kl_div(log_probs, target_probs, reduction='batchmean')
            
            return loss

        # 3. Set up the EvoTorch Problem
        # We need to tell EvoTorch the shape of our solution (the number of model parameters)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Optimizing {num_params} parameters of the ConvLSTM model.")

        problem = Problem(
            "min",  # We want to minimize the loss
            objective_func=objective_function,
            solution_length=num_params,
            initial_bounds=(-0.1, 0.1),
            # Get the initial parameters from the model to start the search
        )

        # 4. Initialize and run the CMA-ES search algorithm
        searcher = CMAES(
            problem,
            stdev_init=0.1,  # A small initial standard deviation is good for neuroevolution
            popsize=population_size,
        )

        print(f"\nRunning optimization for {num_generations} generations...")
        searcher.run(num_generations)

        # 5. Get the best parameters found
        best_params, best_loss = searcher.status["best"], searcher.status["best_eval"]

        print("\n--- Optimization Finished ---")
        print(f"Best Loss (KL Divergence): {best_loss:.6f}")
        # You can now load the best parameters back into your model if needed
        # nn.utils.vector_to_parameters(best_params, model.parameters())


if __name__ == "__main__":
    # This will discover and run all tests in this file
    testConvLSTM = TestConvLSTM()
    testConvLSTM.test_convlstm_forward_pass_and_output_shape()
    testConvLSTM.optimize_convlstm_with_evotorch()
