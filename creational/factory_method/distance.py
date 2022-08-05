# -*- coding: utf-8 -*-
"""This module offers the distance interface and different implementations.

Created on: 5/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class Distance(ABC):
    """Distance interface.

     Declare the common operation of all concrete distances must implement.
     """

    @abstractmethod
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate the distance between two points.

        Args:
            x: data point.
            y: data point.

        Returns:
            dist: distance between point x and y.
        """
        pass


class EuclideanDistance(Distance):
    """Euclidian distance."""

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate the euclidian distance between two points.

        Formula:
            d = [sum{(x_i – y_i)^2}]^(1/2)

        Args:
            x: data point.
            y: data point.

        Returns:
            dist: distance between point x and y.
        """
        difference = x - y

        sum_sq = np.dot(difference.T, difference)

        dist = np.sqrt(sum_sq)

        return dist


class ManhattanDistance(Distance):
    """Cosine distance."""

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate the manhattan distance between two points.

        Args:
            x: data point.
            y: data point.

        Returns:
            dist: distance between point x and y.
        """
        dist = sum(abs(val1-val2) for val1, val2 in zip(x, y))

        return dist
