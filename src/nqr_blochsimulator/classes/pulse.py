import numpy as np


class PulseArray:
    """A class to represent a pulsearray for a NQR sequence."""

    def __init__(self, pulseamplitude, pulsephase, dwell_time) -> None:
        """
        Constructs all the necessary attributes for the pulsearray object.

        Parameters
        ----------
            pulseamplitude : float
                The amplitude of the pulse.
            pulsephase : float
                The phase of the pulse.
            dwell_time : float
                The dwell time of the pulse.
        """
        self.pulseamplitude = pulseamplitude
        self.pulsephase = pulsephase
        self.dwell_time = dwell_time

    def get_real_pulsepower(self) -> np.array:
        """Returns the real part of the pulse power."""
        return self.pulseamplitude * np.cos(self.pulsephase)

    def get_imag_pulsepower(self) -> np.array:
        """Returns the imaginary part of the pulse power."""
        return self.pulseamplitude * np.sin(self.pulsephase)

    @property
    def pulseamplitude(self) -> np.array:
        """Amplitude of the pulse."""
        return self._pulseamplitude

    @pulseamplitude.setter
    def pulseamplitude(self, pulseamplitude):
        self._pulseamplitude = pulseamplitude

    @property
    def pulsephase(self) -> np.array:
        """Phase of the pulse."""
        return self._pulsephase

    @pulsephase.setter
    def pulsephase(self, pulsephase):
        self._pulsephase = pulsephase
