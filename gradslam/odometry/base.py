from abc import ABC, abstractmethod

__all__ = ["OdometryProvider"]


class OdometryProvider(ABC):
    r"""Base class for all odometry providers.

    Your providers should also subclass this class. You should override the `provide()` method.
    """

    def __init__(self, *params):
        r"""Initializes internal OdometryProvider state"""
        pass

    @abstractmethod
    def provide(self, *args, **kwargs):
        r"""Defines the odometry computation performed at every `.provide()` call. """
        raise NotImplementedError
