# -*- coding: utf-8 -*-
"""This module contains the adaptee class.

A fake useful behavior with an incompatible interface with our business logic.

Created on: 28/9/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from typing import Dict


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

    def predict(self, data: Dict[str, float]) -> float:
        """Generate a prediction.

        Args:
            data: Inference data.

        Returns:
            prediction: data predicted.
        """
        y = data.get('y', None)

        if not y:
            raise ValueError("Inferece data is incorrect.")

        return self.slope * y +self.intercept


