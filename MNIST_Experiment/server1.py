from collections import OrderedDict

import flwr as fl
import numpy as np
import torch

from centralized import SimpleCNN

# Initialize the global model
global_model = SimpleCNN()


# Custom FedAvg strategy that saves the global model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list,
    ):
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call base FedAvg aggregate_fit to get aggregated parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated model...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert to PyTorch `state_dict`
            params_dict = zip(global_model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            # Load aggregated model weights
            global_model.load_state_dict(state_dict, strict=True)

            # Save model checkpoint for this round
            model_path = f"global_model_round_{server_round}.pth"
            torch.save(global_model.state_dict(), model_path)
            print(f"Model checkpoint saved: {model_path}")

        return aggregated_parameters, aggregated_metrics


# Start the Flower server with the custom strategy
strategy = SaveModelStrategy()

#starting the server, ie this is where FL starts.
fl.server.start_server(
    server_address="127.0.0.1:8000",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)

print("Federated Learning Complete. Saving final model...")
torch.save(global_model.state_dict(), "global_FL_model.pth")
print("Final global model saved to global_model.pth")
#Saving of model complete