import math

from exceptions.not_instantiable_error import NotInstantiableError


class StatisticsUtils:
    """
    Utility helpers for simple streaming statistics (mean, variance, stddev). This class is not meant to be instantiated.
    """


    ###################
    ### Constructor ###
    ###################
    def __init__(self) -> None:
        """
        Raises:
            NotInstantiableError: Always raised when initialized to indicate that this class is not meant to be instantiated.
        """

        raise NotInstantiableError(f"The class {repr(self.__class__.__name__)} cannot be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def mean(values: list) -> float:
        """
        Computes the arithmetic mean.

        Args:
            values (list): Values to aggregate.

        Returns:
            float: Mean value or NaN if empty.
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
        Computes the empirical variance.

        Args:
            values (list): Values to aggregate.

        Returns:
            float: Variance or NaN if empty.
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
        Computes the empirical standard deviation.

        Args:
            values (list): Values to aggregate.

        Returns:
            float: Standard deviation or NaN if empty.
        """

        var = StatisticsUtils.empirical_variance(values)
        return math.sqrt(var) if not math.isnan(var) else float("nan")
