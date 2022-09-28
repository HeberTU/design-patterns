# -*- coding: utf-8 -*-
"""This module contains the adaptee class.

A fake useful behavior with an incompatible interface with our business logic.

Created on: 28/9/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Dict, List


class LinearModel:
    """Linear model"""

    def __init__(self, slope, intercept):
        """Initialize a lienar model instance.

        Args:
            slope: slope of the model.
            intercept: intercept of the model.
        """
        self.slope = slope
        self.intercept = intercept

    def predict(self, data: Dict[str, List[float]]) -> float:
        """Generate a prediction.

        Args:
            data: Inference data.

        Returns:
            prediction: data predicted.
        """
        Y = data.get('y', None)

        if not Y:
            raise ValueError("Inferece data is incorrect.")

        return [self.slope * y + self.intercept for y in Y]


