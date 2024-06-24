from typing import TypeVar

T = TypeVar("T")


def flatten(l: list[list[T]]) -> list[T]:
    """
    Flatten a list of lists.

    Parameters:
    l   : list of lists

    Returns:
    list with all elements of l in a 1D list
    """
    return [item for sublist in l for item in sublist]
