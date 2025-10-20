import torch
import torch.nn as nn
from evotorch import Problem
from evotorch.algorithms import CMAES

"""
Define a Convolutional LSTM neural network module and a small test/optimization
harness that demonstrates a forward pass and how to optimize the flattened model
parameters with EvoTorch/CMA-ES.

Notes on changes made:
- Removed undefined/unused attributes and methods (self.parsing, tokens, samplesx, samplesy, dist, etc.)
- Fixed JointProbability to accept model outputs and targets and return (log_probs, probs)
  suitable for nn.functional.kl_div (expects log-probabilities as input and probabilities as target).
- Corrected spelling and removed unused `probablity_dist`.
- Simplified the test/optimization flow so methods are self-contained and robust.
- Kept behavior minimal so the file can be run as a script without external dependencies beyond evotorch and torch.
"""


class ConvLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList()
        # first conv: input channels -> hidden_size
        self.conv_layers.append(nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1))
        # subsequent conv layers: hidden_size -> hidden_size
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1))

        # LSTM that consumes a per-time-step feature vector of size hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, C, H, W)
        returns: (batch_size, seq_len, hidden_size)
        """
        b, t, c, h, w = x.size()

        conv_outputs = []
        for time_step in range(t):
            x_t = x[:, time_step, :, :, :]  # (b, c, h, w)
            # apply conv layers
            for conv in self.conv_layers:
                x_t = conv(x_t)
                x_t = torch.relu(x_t)
            # global average pooling to get (b, hidden_size)
            x_t = x_t.mean(dim=[2, 3])
            conv_outputs.append(x_t)

        lstm_input = torch.stack(conv_outputs, dim=1)  # (b, t, hidden_size)
        lstm_out, _ = self.lstm(lstm_input)
        return lstm_out


class TestConvLSTM:
    def JointProbability(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Convert raw outputs and targets to (log_probs, probs) suitable for KL divergence.

        - outputs: logits or raw scores, shape (..., classes)
        - targets: logits or raw scores, shape (..., classes)

        Returns:
        - log_probs: log-softmax(outputs)
        - probs: softmax(targets)
        """
        # Convert outputs to log-probabilities and targets to probabilities.
        log_probs = nn.functional.log_softmax(outputs, dim=-1)
        probs = nn.functional.softmax(targets, dim=-1)
        return log_probs, probs

    def test_convlstm_forward_pass_and_output_shape(self):
        """Tests the forward pass and the output shape of the ConvLSTM."""
        input_size = 3  # Example: RGB channels
        hidden_size = 16
        num_layers = 2
        batch_size = 4
        seq_len = 10
        h, w = 32, 32

        input_tensor = torch.randn(batch_size, seq_len, input_size, h, w)
        model = ConvLSTM(input_size, hidden_size, num_layers)
        output = model.forward(input_tensor)

        # expected shape: (batch_size, seq_len, hidden_size)
        expected_shape = (batch_size, seq_len, hidden_size)
        assert output.shape == expected_shape, f"Got {output.shape}, expected {expected_shape}"
        return output

    def optimize_convlstm_with_evotorch(self, num_generations: int = 10, population_size: int = 5):
        """
        Optimizes the parameters of the ConvLSTM model using EvoTorch and CMA-ES.
        This is a small demo — it optimizes parameters to minimize a KL divergence
        between model outputs and a (here synthetic) target distribution.

        Notes:
        - The target in this demo is a uniform distribution (since target logits are zeros).
        - EvoTorch's objective will be called with a flat parameter vector. We map that
          vector into the model using nn.utils.vector_to_parameters.
        """
        print("\n--- Starting Neuroevolution with EvoTorch and CMA-ES ---")

        # model/data parameters
        input_size, hidden_size, num_layers = 3, 16, 2
        batch_size, seq_len, h, w = 4, 10, 32, 32

        model = ConvLSTM(input_size, hidden_size, num_layers)

        # sample input and (synthetic) target logits
        sample_input = torch.randn(batch_size, seq_len, input_size, h, w)
        target_output = torch.zeros(batch_size, seq_len, hidden_size)  # logits -> uniform after softmax

        # count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Optimizing {num_params} parameters of the ConvLSTM model.")

        # objective to minimize
        def objective_function(params: torch.Tensor):
            # params: a 1D tensor of length num_params
            if params.numel() != num_params:
                raise ValueError(f"Parameter vector has wrong size: {params.numel()} != {num_params}")

            # load flat params into model
            nn.utils.vector_to_parameters(params, model.parameters())

            # forward pass
            output = model.forward(sample_input)  # (batch_size, seq_len, hidden_size)

            # reshape to (batch_size * seq_len, hidden_size)
            output_reshaped = output.reshape(batch_size * seq_len, hidden_size)
            target_reshaped = target_output.reshape(batch_size * seq_len, hidden_size)

            # produce proper (log_probs, probs) for KLDivLoss
            log_probs, target_probs = self.JointProbability(output_reshaped, target_reshaped)

            # KLDivLoss expects input=log_probs, target=probs
            loss = nn.functional.kl_div(log_probs, target_probs, reduction="batchmean")
            # CMA-ES minimizes the returned scalar; ensure it's detached and a 0-dim tensor
            return loss

        # set up EvoTorch problem
        problem = Problem(
            "min",
            objective_func=objective_function,
            solution_length=num_params,
            initial_bounds=(-0.1, 0.1),
        )

        # set up and run CMA-ES
        searcher = CMAES(
            problem,
            stdev_init=0.1,
            popsize=population_size,
        )

        print(f"\nRunning optimization for {num_generations} generations...")
        searcher.run(num_generations)

        best_params, best_loss = searcher.status["best"], searcher.status["best_eval"]

        print("\n--- Optimization Finished ---")
        print(f"Best Loss (KL Divergence): {best_loss:.6f}")
        # Optionally load best parameters back:
        # nn.utils.vector_to_parameters(best_params, model.parameters())

        return best_params, best_loss


if __name__ == "__main__":
    tester = TestConvLSTM()
    print("Testing forward pass...")
    tester.test_convlstm_forward_pass_and_output_shape()
    print("Running EvoTorch optimization demo (this will take time depending on generations/popsize)...")
    tester.optimize_convlstm_with_evotorch()
