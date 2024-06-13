from typing import Any, Callable, Iterator, Tuple

import numpy as np
import tensorflow as tf


def random_crop(
    x: np.ndarray, y: np.ndarray, img_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    stacked_image = tf.stack([x, y], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, img_size, img_size, 1])

    return cropped_image[0], cropped_image[1]


def resize(
    x: np.ndarray, y: np.ndarray, img_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    x = tf.image.resize(
        x, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    y = tf.image.resize(
        y, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return x, y


@tf.function()
def random_jitter(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Resizing to 286x286
    x, y = resize(x, y, 286)

    # Random cropping back to 256x256
    x, y = random_crop(x, y, 256)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)

    return x, y


def sample_norm_01_full_dataset(x: np.ndarray, type) -> np.ndarray:
    if type == "minmax":
        x = x.astype(np.float32)
        n_x: np.ndarray = (x - np.amin(x, axis=(1, 2), keepdims=True)) / (
            np.amax(x, axis=(1, 2), keepdims=True)
            - np.amin(x, axis=(1, 2), keepdims=True)
        )

        return n_x * 2 - 1
    if type == "const":
        return (x / 127.5) - 1
    else:
        assert False


class NpyDataloader:
    def __init__(self, path: str, im_size: int, random_jitter: bool, norm_type: str) -> None:
        self._x = sample_norm_01_full_dataset(
            np.load(f"{path}/x.npy", mmap_mode="r+"), norm_type
        )
        self._y = sample_norm_01_full_dataset(
            np.load(f"{path}/y.npy", mmap_mode="r+"), norm_type
        )
        print("Data shape:", self._x.shape)
        self._im_size = im_size
        self._random_jitter = random_jitter
        self._type = norm_type

    def __len__(self) -> int:
        len = self._x.shape[0]
        assert isinstance(len, int)
        return len

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self._x[idx], self._y[idx]

        if x.shape[0] > self._im_size or x.shape[1] > self._im_size:
            center_x = np.random.randint(
                self._im_size // 2, x.shape[1] - self._im_size // 2
            )
            center_y = np.random.randint(
                self._im_size // 2, x.shape[0] - self._im_size // 2
            )

            x = x[
                center_y - self._im_size // 2 : center_y + self._im_size // 2,
                center_x - self._im_size // 2 : center_x + self._im_size // 2,
            ]
            y = y[
                center_y - self._im_size // 2 : center_y + self._im_size // 2,
                center_x - self._im_size // 2 : center_x + self._im_size // 2,
            ]

        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
            y = np.expand_dims(y, -1)

        if self._random_jitter:
            x, y = random_jitter(x, y)

        return x, y

    def __call__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def prepare_dataset(
    path: str, im_size: int, random_jitter: bool = False, norm_type="minmax"
) -> Tuple[tf.data.Dataset, tf.TensorShape]:
    dataloader: Callable[
        [], Iterator[Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]
    ]

    dataloader = NpyDataloader(
        path=path, im_size=im_size, random_jitter=random_jitter, norm_type=norm_type
    )
    channels = 1

    shape = tf.TensorShape([im_size, im_size, channels])
    dataset = tf.data.Dataset.from_generator(
        dataloader,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            shape,
            shape,
        ),
    )
    return dataset, shape
