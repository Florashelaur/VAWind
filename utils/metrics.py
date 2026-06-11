import numpy as np


def metric(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    error = predictions - targets
    absolute_error = np.abs(error)
    epsilon = 1e-8

    return {
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(
            np.mean(absolute_error / np.maximum(np.abs(targets), epsilon))
        ),
        "smape": float(
            200.0
            * np.mean(
                absolute_error
                / np.maximum(
                    np.abs(targets) + np.abs(predictions), epsilon
                )
            )
        ),
    }
