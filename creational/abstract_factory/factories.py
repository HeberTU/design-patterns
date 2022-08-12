# -*- coding: utf-8 -*-
"""Factory module.

Created on: 12/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from creational.abstract_factory import vectors


class AbstractFactory(ABC):
    """
    The Abstract Factory interface declares a set of methods that return
    different abstract products.
    """
    @abstractmethod
    def create_vector(self) -> vectors.Vector:
        """Create a vector instance."""
        pass


class NumpyFactory(AbstractFactory):
    """Numpy factory."""

    def create_vector(self, *args) -> vectors.NumpyVector:
        """Create an instance of one-dimensional vector.

        Returns:
            vector: one-dimensional vector.
        """
        return vectors.NumpyVector(args)


class TorchFactory(AbstractFactory):
    """Torch factory."""

    def create_vector(self, *args) -> vectors.TorchVector:
        """Create an instance of one-dimensional vector.

        Returns:
            vector: one-dimensional vector.
        """
        return vectors.TorchVector(args)
