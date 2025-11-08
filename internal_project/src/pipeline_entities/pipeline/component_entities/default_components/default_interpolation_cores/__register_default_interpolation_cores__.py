### DO NOT REMOVE IMPORTS, THEY ARE NECESSARY!  ###


def register():
    """
    Register the interpolation cores.
    """

    from . import newton_interpolation_core
    from . import barycentric_first_interpolation_core
    from . import barycentric_second_interpolation_core
    from . import barycentric_second_chebyshev_interpolation_core
    from . import aitken_neville_interpolation_core
    from . import chebyshev_interpolation_matrix_core
