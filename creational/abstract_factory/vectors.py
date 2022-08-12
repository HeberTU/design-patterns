# -*- coding: utf-8 -*-
"""Vectors module

Created on: 11/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations
from typing import Iterable, Union, List
from abc import ABC, abstractmethod

import torch
import numpy as np


class Vector(ABC):
    """Vector interface"""

    @abstractmethod
    def dot_product(self, other: Vector):
        """Perform dot product between two vectors."""

    @property
    def data(self) -> List[Union[float, int]]:
        """Get vector data."""
        return list(self._data)

    def __repr__(self) -> str:
        """Get human-readable representation"""
        return f"{self.__class__.__name__}({self.data})"


class NumpyVector(Vector):
    """Regular nupy one-dimentional vector."""

    def __init__(
        self, *args: Iterable[Union[int, float]]
    ) -> NumpyVector:
        """Instantiate a Numpy on dimensional vector.

        Args:
            *args: Numeric iterable from which the vector will be created.
        """
        self._data = np.array(args)

    def dot_product(self, other: NumpyVector) -> float:
        """Perform dot product between two one-dimensional vectors.

        Args:
            other: the other numpy vector.

        Returns:
            dot_prod: resulting dot product.
        """
        return sum(self._data * self._data)


class TorchVector(Vector):
    """Torch one-dimentional vector."""
    def __init__(
            self, *args: Iterable[Union[int, float]]
    ) -> TorchVector:
        """Initialize a torch one dimensional vector.

        Args:
            *args: Numeric iterable from which the vector will be created.
        """
        self._data = torch.Tensor(args)

    def dot_product(self, other: TorchVector) -> float:
        """Perform dot product between two one-dimensional vectors.

        Args:
            other: the other numpy vector.

        Returns:
            dot_prod: resulting dot product.
        """
        return sum(self._data * self._data)

    @property
    def data(self) -> List[float]:
        """Get vector data."""
        return [float(x) for x in self._data]
