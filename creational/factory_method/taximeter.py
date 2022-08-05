# -*- coding: utf-8 -*-
"""This module has the taximeter interface and implementations.

Created on: 5/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from creational.factory_method import distance


class Taximeter(ABC):
    """Taximeter class."""

    @abstractmethod
    def factory_method(self) -> distance.Distance:
        """Instantiate the right distance."""
        pass

    def estimate_cost(self, x: np.ndarray, y: np.array) -> float:
        """Estimate ride cost.

        Args:
            x: start point.
            y: finish point.

        Returns:
            cost: ride cost.
        """
        distance_calculator = self.factory_method()
        dist = distance_calculator.calculate(x, y)

        cost = 1.27 * dist

        return cost


class CityTaximeter(Taximeter):
    """City Taximeter.

    Calculate ride cost based on manhattan distance.
    """

    def factory_method(self) -> distance.ManhattanDistance:
        """Instantiate the manhattan distance."""
        return distance.ManhattanDistance()


class RuralTaximeter(Taximeter):
    """Rural Taximeter.

    Calculate ride cost based on euclidian distance.
    """

    def factory_method(self) -> distance.EuclideanDistance:
        """Instantiate the manhattan distance."""
        return distance.EuclideanDistance()
