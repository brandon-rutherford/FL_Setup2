import argparse
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch

from centralized import UNet
from StartConfig import *

# Initialize the global model
global_model = UNet(in_channels=1, out_channels=1)

def aggregate_evaluate_metrics(
    results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]]
):
    """Aggregate evaluation metrics by computing a weighted average."""
    total_loss = 0.0
    total_examples = 0
    dice_sum = 0.0
    pixel_accuracy_sum = 0.0

    for _, evaluate_res in results:
        n = evaluate_res.num_examples
        total_loss += evaluate_res.loss * n
        total_examples += n
        metrics = evaluate_res.metrics or {}
        dice_sum += metrics.get("DICE Score", 0.0) * n
        pixel_accuracy_sum += metrics.get("Pixel Accuracy", 0.0) * n

    aggregated_loss = total_loss / total_examples if total_examples > 0 else 0.0
    aggregated_metrics = {
        "DICE Score": dice_sum / total_examples if total_examples > 0 else 0.0,
        "Pixel Accuracy": pixel_accuracy_sum / total_examples if total_examples > 0 else 0.0,
    }
    return aggregated_loss, aggregated_metrics

# Custom FedAvg strategy with global model saving, evaluation, and early stopping.
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_dice = 0.0
        self.no_improvement_rounds = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list,
    ):
        """Aggregate model weights using weighted average and store checkpoint."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated model...")
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(global_model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            global_model.load_state_dict(state_dict, strict=True)
            model_path = (
                f"{NUM_CLIENTS},{NUM_EPOCHS},{EPOCH_EARLY_STOPPING_PATIENCE},{BATCH_SIZE},"
                f"{NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP},{CLIENT_PARTICIPATION_FRACTION},{NUM_ROUNDS},"
                f"GlobalModel:{server_round}.pth"
            )
            torch.save(global_model.state_dict(), model_path)
            print(f"Model checkpoint saved: {model_path}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: list,
    ):
        aggregated_loss, aggregated_metrics = aggregate_evaluate_metrics(results)
        if aggregated_metrics is not None and "DICE Score" in aggregated_metrics:
            current_dice = aggregated_metrics["DICE Score"]
            print(f"Round {server_round} evaluation metrics: {aggregated_metrics}")
            if current_dice > self.best_dice:
                self.best_dice = current_dice
                self.no_improvement_rounds = 0
            else:
                self.no_improvement_rounds += 1
        else:
            print(f"Round {server_round}: No DICE Score available in aggregated metrics.")
        return aggregated_loss, aggregated_metrics

    def should_stop(self, server_round: int) -> bool:
        if (
            self.no_improvement_rounds >= NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP
            and NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP != 0
        ):
            print(f"Early stopping: Global DICE did not improve for {NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP} consecutive rounds.")
            return True
        return False


if __name__ == "__main__":

    strategy = SaveModelStrategy(
        fraction_fit=CLIENT_PARTICIPATION_FRACTION,
        fraction_evaluate=CLIENT_PARTICIPATION_FRACTION,
        min_evaluate_clients=1,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        fit_metrics_aggregation_fn=lambda results: {},  # Dummy function to silence warning
        #min_available_clients=NUM_CLIENTS,
        min_fit_clients=int(NUM_CLIENTS*CLIENT_PARTICIPATION_FRACTION),
    )

    fl.server.start_server(
        server_address="127.0.0.1:8000",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("Federated Learning Complete. Saving final model...")
    torch.save(
        global_model.state_dict(),
        f"{NUM_CLIENTS},{NUM_EPOCHS},{EPOCH_EARLY_STOPPING_PATIENCE},{BATCH_SIZE},{NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP},"
        f"{CLIENT_PARTICIPATION_FRACTION},{NUM_ROUNDS},Global.pth"
    )
    print(
        f"Final global model saved to {NUM_CLIENTS},{NUM_EPOCHS},{EPOCH_EARLY_STOPPING_PATIENCE},{BATCH_SIZE},"
        f"{NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP},{CLIENT_PARTICIPATION_FRACTION},{NUM_ROUNDS},Global.pth"
    )
