from typing import List, Optional
from ..data_basic import Dataset
import gzip
import struct
import numpy as np


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        with gzip.open(image_filename, "rb") as file:
            magic, num_images, rows, cols = struct.unpack(
                ">IIII", file.read(16)
            )
            self.X = np.frombuffer(
                file.read(), dtype=np.uint8
            ).reshape(num_images, rows * cols)
            self.X = self.X.astype(np.float32) / 255.0

        with gzip.open(label_filename, "rb") as file:
            label_magic, num_labels = struct.unpack(
                ">II", file.read(8)
            )
            self.y = np.frombuffer(
                file.read(), dtype=np.uint8
            )

        if num_images != num_labels:
            raise ValueError(
                "Image and label counts do not match"
            )

        self.rows = rows
        self.cols = cols
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images = self.X[index]
        labels = self.y[index]

        if np.isscalar(index):
            image = images.reshape(
                self.rows, self.cols, 1
            )
            return self.apply_transforms(image), labels

        transformed = []
        for image in images:
            image = image.reshape(
                self.rows, self.cols, 1
            )
            transformed.append(
                self.apply_transforms(image)
            )

        return np.stack(transformed), labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION
