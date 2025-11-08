### DO NOT REMOVE IMPORTS, THEY ARE NECESSARY!  ###


def register():
    """
    Register the evaluation components.
    """

    from . import interpolation_values_evaluator
    from . import interpolant_evaluator
    from . import aitken_neville_evaluator
