# -*- coding: utf-8 -*-
"""This module contains the client code.

Created on: 28/9/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import pandas as pd
from random import random
from structural.adapter.linear_model import LinearModel


class Adapter:
    """Domain-specific interface used by the client code."""

    def __init__(self, model: LinearModel):
        """Instantiate an adapter object."""
        self.model = model

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use the model attribute to generate predictions

        Args:
            df: data frame containing the sensor data.

        Returns:
            df: data frame containing the transformed sensor data.
        """
        data = df.to_dict()
        data['y'] = data.pop('meassure')

        return df.assign(
            prediction=self.model.predict(data=df)
        )


def get_sensor_data(n: int = 5) -> pd.DataFrame:
    """Get data from a fake sensor.
    Args:
        n: number of samples.
    Returns:
        df: data frame containing the sensor data.
    """
    df = pd.DataFrame(
        data={'meassure': [random() for i in range(n)]}
    )

    return df


def assign_action(df: pd.DataFrame, adapter: Adapter) -> pd.DataFrame:
    """Assign an action base on the predicted value.

    Args:
        df: data frame containing the sensor data.

    Returns:
        df: data frame containing the sensor data with their assigned actions.
    """
    df = adapter.predict(df)
    df['action'] = df['prediction'].apply(lambda x: 0 if x<1 else 1)

    return df
