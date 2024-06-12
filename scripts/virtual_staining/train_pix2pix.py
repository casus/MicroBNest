import argparse
from collections import defaultdict

import neptune as neptune
import numpy as np
import tensorflow as tf
from cvdm.architectures.pix2pix import Discriminator, Generator
from cvdm.utils.metrics_utils import calculate_metrics
from cvdm.utils.training_utils import prepare_dataset
from tqdm import tqdm

tf.keras.utils.set_random_seed(42)


def generator_loss(discriminator_generated_output, generator_output, ground_truth):
    gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(discriminator_generated_output), discriminator_generated_output
    )
    l1_loss = tf.reduce_mean(tf.abs(ground_truth - generator_output))
    total_loss = gen_loss + 100 * l1_loss
    return total_loss, gen_loss, l1_loss


def discriminator_loss(
    discriminator_real_output, discriminator_generated_output, disc_weight
):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(discriminator_real_output), discriminator_real_output
    )
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(discriminator_generated_output), discriminator_generated_output
    )
    total_loss = real_loss + generated_loss
    return total_loss * disc_weight


@tf.function
def train_step(
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    input_image,
    target,
    disc_weight,
):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_total_loss = discriminator_loss(
            disc_real_output, disc_generated_output, disc_weight
        )

    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_total_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )
    return gen_output, gen_total_loss, gen_loss, gen_l1_loss, disc_total_loss


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
    discriminator = Discriminator()
    generator.summary()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    print("Starting training...")

    epochs = 400
    best_val_mse = 1000000

    for _ in tqdm(range(epochs)):
        cumulative_loss = 0.0
        train_metrics = defaultdict(float)
        for batch in dataset:
            batch_x, batch_y = batch
            output, gen_total_loss, gen_loss, gen_l1_loss, disc_total_loss = train_step(
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                batch_x,
                batch_y,
                0.5,
            )
            cumulative_loss += np.array(
                [gen_total_loss, gen_loss, gen_l1_loss, disc_total_loss]
            )
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
