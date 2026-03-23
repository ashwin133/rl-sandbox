from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class BasePlotter(ABC):
    """Base class for plotting data."""

    @abstractmethod
    def plot(self, frame: np.ndarray) -> None:
        """Plot a single frame of data."""
        ...


class MatplotlibPlotter(BasePlotter):
    """Plotter that uses Matplotlib to render frames."""

    def plot(self, frame: np.ndarray) -> None:
        """Plot a single frame of data using Matplotlib."""
        plt.imshow(frame)
        plt.axis("off")
        plt.show()
