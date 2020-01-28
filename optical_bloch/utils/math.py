def anti_commute(A,B):
    """
    Calculate the anti-commutator between A and B; AB+BA.

    Parameters:
    A   : 2D array or matrix
    B   : 2D array or matrix

    Returns:
    2D array or matrix AB+BA
    """
    return A@B+B@A

def commute(A,B):
    """
    Calculate the commutator between A and B; AB-BA.

    Parameters:
    A   : 2D array or matrix
    B   : 2D array or matrix

    Returns:
    2D array or matrix AB+BA
    """
    return A@B-B@A
