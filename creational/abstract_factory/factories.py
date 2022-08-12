# -*- coding: utf-8 -*-
"""Factory module.

Created on: 12/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Tuple, Union, List

from creational.abstract_factory import (
    vectors,
    matrices
)


class AbstractFactory(ABC):
    """
    The Abstract Factory interface declares a set of methods that return
    different abstract products.
    """
    @abstractmethod
    def create_vector(self) -> vectors.Vector:
        """Create a vector instance."""
        pass

    @abstractmethod
    def create_matrix(
        self, shape: Tuple[int, int], data: List[Union[int, float]]
    ) -> matrices.Matrix:
        """Create a matrix instance."""
        pass


class NumpyFactory(AbstractFactory):
    """Numpy factory."""

    def create_vector(self, *args) -> vectors.NumpyVector:
        """Create an instance of one-dimensional vector.

        Returns:
            vector: one-dimensional vector.
        """
        return vectors.NumpyVector(args)

    def create_matrix(
        self, shape: Tuple[int, int], data: List[Union[int, float]]
    ) -> matrices.NumpyMatrix:
        """Create an instance of numpy matrix.

        Args:
            shape: matrix shape.
            data: data to feed the matrix.

        Returns:

        """
        return matrices.NumpyMatrix(shape, data)


class TorchFactory(AbstractFactory):
    """Torch factory."""

    def create_vector(self, *args) -> vectors.TorchVector:
        """Create an instance of one-dimensional vector.

        Returns:
            vector: one-dimensional vector.
        """
        return vectors.TorchVector(args)

    def create_matrix(
        self, shape: Tuple[int, int], data: List[Union[int, float]]
    ) -> matrices.TensorMatrix:
        """Create an instance of tensor matrix.

        Args:
            shape: matrix shape.
            data: data to feed the matrix.

        Returns:

        """
        return matrices.TensorMatrix(shape, data)
