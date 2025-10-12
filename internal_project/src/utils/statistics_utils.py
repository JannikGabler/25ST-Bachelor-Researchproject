import math

from exceptions.not_instantiable_error import NotInstantiableError


class StatisticsUtils:

    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        raise NotInstantiableError(
            f"The class {repr(self.__class__.__name__)} cannot be instantiated."
        )

    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def mean(values: list) -> float:
        """
        TODO
        """

        n = 0
        mean = 0.0
        for x in values:
            n += 1
            mean += (x - mean) / n
        return mean if n > 0 else float("nan")

    @staticmethod
    def empirical_variance(values: list) -> float:
        """
        TODO
        """

        n = 0
        mean = 0.0
        M2 = 0.0
        for x in values:
            n += 1
            delta = x - mean
            mean += delta / n
            M2 += delta * (x - mean)
        return M2 / n if n > 0 else float("nan")

    @classmethod
    def empirical_stddev(cls, values: list) -> float:
        """
        TODO
        """

        var = StatisticsUtils.empirical_variance(values)
        return math.sqrt(var) if not math.isnan(var) else float("nan")
