# -*- coding: utf-8 -*-
"""Matrix vector.

Created on: 12/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations
from typing import List, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import torch


class Matrix(ABC):
    """Matrix interface."""

    @abstractmethod
    def dot_product(self, other: Matrix):
        """Perform dot product between two matrices."""
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape"""
        return self._shape

    @property
    def data(self) -> List[List[float]]:
        """Get matrix data."""
        return [
            list(x)
            for x in self._data
        ]

    def __repr__(self) -> str:
        """Get human-readable representation"""
        return f"{self.__class__.__name__}({self.data})"


def _shape_data(
        shape: Tuple[int, int], data: List[Union[int, float]]
) -> List[List[float]]:
    """Shape the data into a matrix format.

    Args:
        shape: matrix shape.
        data: data to feed the matrix.

    Returns:
        _data: data in the right format.
    """
    if shape[0] * shape[1] != len(data):
        raise ValueError("Shape is not compatible.")

    _data = []

    for i in range(shape[0]):
        _data.append(data[i: i + shape[1]])

    return _data


class NumpyMatrix(Matrix):
    """Numpy matrix interface."""

    def __init__(
        self, shape: Tuple[int, int], data: List[Union[int, float]]
    ):
        """Initialize a 2-dimensional array..

        Args:
            shape: matrix shape.
            data: data to feed the matrix.
        """
        self._shape = shape

        self._data = np.array(_shape_data(shape, data))

    def dot_product(self, other: NumpyMatrix):
        """Perform dot product between two matrices."""
        return np.dot(self._data, other._data)


class TensorMatrix(Matrix):
    """Tensor matrix interface."""

    def __init__(
        self, shape: Tuple[int, int], data: List[Union[int, float]]
    ):
        """Initialize a 2-dimensional tensor.

        Args:
            shape: matrix shape.
            data: data to feed the matrix.
        """
        self._shape = shape

        self._data = torch.Tensor(_shape_data(shape, data))

    def dot_product(self, other: TensorMatrix):
        """Perform dot product between two matrices."""
        return torch.matmul(self._data, other._data)
