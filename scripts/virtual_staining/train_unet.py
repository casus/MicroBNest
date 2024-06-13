import argparse
from collections import defaultdict

import neptune as neptune
import tensorflow as tf
from stardist.models.pix2pix import Generator
from stardist.src.models.virtual_staining_metrics import calculate_metrics
from stardist.src.data.virtual_staining_datasets import prepare_dataset
from tqdm import tqdm

tf.keras.utils.set_random_seed(42)


@tf.function
def train_step(
    generator,
    generator_optimizer,
    input_image,
    target,
):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)

        loss = tf.reduce_mean(tf.abs(gen_output - target))

    generator_gradients = gen_tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    return gen_output, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path", help="Path to the configuration file", required=True
    )
    parser.add_argument(
        "--val-path", help="Path to the configuration file", required=True
    )

    args = parser.parse_args()

    print("Num CPUs Available: ", len(tf.config.list_physical_devices("CPU")))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    print("Getting data...")

    dataset, _ = prepare_dataset(
        path=args.train_path,
        im_size=256,
        random_jitter=True,
        norm_type="minmax",
    )
    val_dataset, _ = prepare_dataset(
        path=args.val_path, im_size=256, norm_type="minmax"
    )

    batch_size = 8

    dataset = dataset.shuffle(5000)
    val_dataset = val_dataset.shuffle(5000)

    dataset = dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    generator = Generator()
    generator.summary()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    print("Starting training...")

    epochs = 400
    best_val_mse = 1000000

    for _ in tqdm(range(epochs)):
        cumulative_loss = 0.0
        train_metrics = defaultdict(float)
        for batch in dataset:
            batch_x, batch_y = batch
            output, gen_total_loss = train_step(
                generator,
                generator_optimizer,
                batch_x,
                batch_y,
            )
            cumulative_loss += gen_total_loss
            metrics = calculate_metrics(output, batch_y.numpy())
            for k, v in metrics.items():
                train_metrics[k] += v

        val_metrics = defaultdict(float)
        for n, batch in enumerate(val_dataset):
            batch_x, batch_y = batch
            output = generator(batch_x, training=True)
            metrics = calculate_metrics(output, batch_y.numpy())
            for k, v in metrics.items():
                val_metrics[k] += v

        for k, v in val_metrics.items():
            val_metrics[k] = val_metrics[k] / (n + 1)

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]


if __name__ == "__main__":
    main()
