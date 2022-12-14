# -*- coding: utf-8 -*-
"""Client code.

Created on: 5/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import numpy as np
from creational.factory_method import taximeter


def client_code(
        ride_calculator: taximeter.Taximeter,
        x: np.ndarray,
        y: np.ndarray,
) -> float:
    """Calculate the ride cost

    The client code works with an instance of a concrete taximeter, albeit
    through its base interface. As long as the client keeps working with the
    taximeter via the base interface, you can pass it any taximeter's subclass.

    Args:
        ride_calculator:
        x: start point.
        y: finish point.

    Returns:
        cost: ride cost.
    """
    return ride_calculator.estimate_cost(x, y)


if __name__ == '__main__':
    print("App configuration: City Taximeter.")
    cost = client_code(
        ride_calculator=taximeter.CityTaximeter(),
        x=np.array([0, 0]),
        y=np.array([1, 1]),
    )
    print(cost)

    print("App configuration: Rural Taximeter.")
    cost = client_code(
        ride_calculator=taximeter.RuralTaximeter(),
        x=np.array([0, 0]),
        y=np.array([1, 1]),
    )
    print(cost)
