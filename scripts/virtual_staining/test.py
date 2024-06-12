from collections import defaultdict

import numpy as np
import plotly.io as pio
from cvdm.architectures.pix2pix import Generator
from cvdm.utils.metrics_utils import calculate_metrics
from cvdm.utils.training_utils import prepare_dataset
from tqdm import tqdm

pio.renderers.default = "iframe"
BASE_PATH = "/home/wyrzyk93/DeepStain/outputs/weights/"


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
    path="/bigdata/casus/MLID/maria/cyt_to_nuc/test", im_size=256, norm_type="minmax"
)
test_dataset = test_dataset.batch(8)

# Pix2pix
generator = Generator()
generator.load_weights(f"{BASE_PATH}best_model_f3619e47-0d44-4ba4-aac7-f56ee3ac77fc.h5")

results = calculate_metrics_multiple_times(generator, test_dataset)
print(results)

# UNet
generator = Generator()
generator.load_weights(f"{BASE_PATH}best_model_5a0b8dce-7a9c-4269-9434-16d9a11b9975.h5")

results = calculate_metrics_multiple_times(generator, test_dataset)
print(results)

test_dataset, _ = prepare_dataset(
    path="/bigdata/casus/MLID/maria/nuc_to_cyt/test", im_size=256, norm_type="minmax"
)
test_dataset = test_dataset.batch(8)

# Pix2pix
generator = Generator()
generator.load_weights(f"{BASE_PATH}best_model_bdc72f48-7ffa-4c19-b5e9-ac545d14c90b.h5")

results = calculate_metrics_multiple_times(generator, test_dataset)
print(results)

# UNet
generator = Generator()
generator.load_weights(f"{BASE_PATH}best_model_3a930e03-f8a6-4fa0-964f-2c7a3905afd7.h5")

results = calculate_metrics_multiple_times(generator, test_dataset)
print(results)
