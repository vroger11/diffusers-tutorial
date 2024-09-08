"""Module to create the sprite dataset.

Done to avoid unnecessary import while loading the dataset.
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class SpritesDataset(Dataset):
    """Dataset to load the sprite data for tutorials."""

    def __init__(
        self,
        sfilename: str,
        lfilename: str,
        null_context: bool = False,
        clean_version: bool = False,
    ) -> None:
        """Initialize sprite dataset.

        Parameters
        ----------
        sfilename : str
            filepath to the sprites npy file
        lfilename : str
            filepath to labels associated to the sprites from sfilename file
        null_context : bool, optional
            Set all labels to zero if True, else use the right label, by default False
        """
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # from [0,255] to range [0.0,1.0]
                transforms.Normalize((0.5,), (0.5,)),  # range [-1,1]
            ]
        )
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape

        # clean dataset
        # Note: in real world application this is done before to avoid wasting time cleaning
        if clean_version:
            idx_to_delete = [
                idx for idx, image in enumerate(self.sprites) if detect_wrong_images(image)
            ]

            self.sprites = np.delete(self.sprites, idx_to_delete, axis=0)
            self.slabels = np.delete(self.slabels, idx_to_delete, axis=0)

    # Return the number of images in the dataset
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.sprites)

    # Get the image and label at a given index
    def __getitem__(self, idx: int) -> tuple:
        """Get sample from index `ìdx`.

        Parameters
        ----------
        idx : int
            index

        Returns
        -------
        tuple
            image and label associated to the `idx` sample.
        """
        # Return the image and label as a tuple
        image = self.transform(self.sprites[idx])
        if self.null_context:
            label = torch.tensor(0).to(torch.int64)
        else:
            label = torch.tensor(self.slabels[idx]).to(torch.int64)

        return image, label

    def getshapes(self) -> tuple[tuple[int], tuple[int]]:
        """Return the shape of sprites and the shape of labels.

        Returns
        -------
        tuple[tuple[int], tuple[int]]
            The first tuple is the shape of sprites and the second one is the shape of labels.
        """
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape


def detect_wrong_images(image: np.ndarray, number_of_white: int = 26) -> bool:
    """Function to detect if an image is a wrong one and that it should not be used for training.

    Parameters
    ----------
    image : np.ndarray
        the image in RGB with shape [width, heigh, channel]
    number_of_white : int, optional
        the number of white pixel a function must have to be considered valid, by default 26

    Returns
    -------
    bool
        True if it is a wrong image for generation and False if it is a good one.
    """

    wrong_image = False
    # Note white is (255, 255, 255) hence the sum is 765
    # White background so a minimum of pixel should be white
    wrong_image |= np.sum(image.sum(axis=-1) == 765) < number_of_white

    # print(image == 255)
    # print(np.sum(image == 255))
    # wrong_image |= np.sum(image == 255) < 10
    # 4 corners must be white
    wrong_image |= image[0, 0].sum() != 765
    wrong_image |= image[-1, 0].sum() != 765
    wrong_image |= image[0, -1].sum() != 765
    wrong_image |= image[-1, -1].sum() != 765

    return wrong_image
