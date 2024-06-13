from collections import defaultdict

import numpy as np
import plotly.io as pio
from stardist.models.pix2pix import Generator
from stardist.src.models.virtual_staining_metrics import calculate_metrics
from stardist.src.data.virtual_staining_datasets import prepare_dataset
from tqdm import tqdm

pio.renderers.default = "iframe"
BASE_PATH = ""


def evaluate_model(generator, test_dataset):
    test_metrics = defaultdict(float)
    for n, batch in tqdm(enumerate(test_dataset)):
        batch_x, batch_y = batch
        output = generator(batch_x, training=True)
        metrics = calculate_metrics(output, batch_y.numpy())
        for k, v in metrics.items():
            test_metrics[k] += v

    for k, v in test_metrics.items():
        test_metrics[k] = test_metrics[k] / (n + 1)

    return test_metrics


def calculate_metrics_multiple_times(generator, test_dataset, runs=3):
    all_metrics_runs = []

    for _ in range(runs):
        metrics = evaluate_model(generator, test_dataset)
        all_metrics_runs.append(metrics)

    aggregated_metrics = defaultdict(list)
    for metrics in all_metrics_runs:
        for key, value in metrics.items():
            aggregated_metrics[key].append(value)

    metrics_mean_stdev = {}
    for key, values in aggregated_metrics.items():
        mean_value = np.mean(values)
        stdev_value = np.std(values)
        metrics_mean_stdev[key] = {"mean": mean_value, "stdev": stdev_value}

    return metrics_mean_stdev


test_dataset, _ = prepare_dataset(
    path="", im_size=256, norm_type="minmax"
)
test_dataset = test_dataset.batch(8)

# Pix2pix
generator = Generator()
generator.load_weights(f"")

results = calculate_metrics_multiple_times(generator, test_dataset)
print(results)

# UNet
generator = Generator()
generator.load_weights(f"")

results = calculate_metrics_multiple_times(generator, test_dataset)
print(results)

test_dataset, _ = prepare_dataset(
    path="", im_size=256, norm_type="minmax"
)
test_dataset = test_dataset.batch(8)

# Pix2pix
generator = Generator()
generator.load_weights(f"")

results = calculate_metrics_multiple_times(generator, test_dataset)
print(results)

# UNet
generator = Generator()
generator.load_weights(f"")

results = calculate_metrics_multiple_times(generator, test_dataset)
print(results)
