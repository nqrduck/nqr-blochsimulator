import numpy as np
import logging
from scipy.constants import h, Boltzmann
from .sample import Sample
from .pulse import PulseArray


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class Simulation:
    """Class for the simulation of the Bloch equations."""

    def __init__(
        self,
        sample: Sample,
        number_isochromats: int,
        initial_magnetization: float,
        gradient: float,
        noise: float,
        length_coil: float,
        diameter_coil: float,
        number_turns: float,
        q_factor_transmit: float,
        q_factor_receive: float,
        power_amplifier_power: float,
        pulse: PulseArray,
        averages: int,
        gain: float,
        temperature: float,
        loss_TX: float = 0,
        loss_RX: float = 0,
        conversion_factor: float = 1,
    ) -> None:
        """
        Constructs all the necessary attributes for the simulation object.

        Parameters
        ----------
            sample : Sample
                The sample that is used for the simulation.
            number_isochromats : int
                The number of isochromats used for the simulation.
            initial_magnetization : float
                The initial magnetization of the sample.
            gradient : float
                The gradient of the magnetic field in mt/M.
            noise : float
                The RMS Noise of the measurement setup in µVolts.
            length_coil : float
                The length of the coil in meters.
            diameter_coil : float
                The diameter of the coil in meters.
            number_turns : float
                The number of turns of the coil.
            q_factor_transmit : float
                The Q-factor of the transmit path of the probe coil.
            q_factor_receive : float
                The Q-factor of the receive path of the probe coil.
            power_amplifier_power : float
                The power of the power amplifier in Watts.
            pulse: PulseArray
                The pulse that is used for the simulation.
            averages:
                The number of averages that are used for the simulation.
            gain:
                The gain of the amplifier.
            temperature:
                The temperature of the sample in Kelvin.
            loss_TX:
                The loss of the transmitter in dB.
            loss_RX:
                The loss of the receiver in dB.
            conversion_factor:
                The conversion factor of the receiver in spectromter units / Volt.

        """
        self.sample = sample
        self.number_isochromats = number_isochromats
        self.initial_magnetization = initial_magnetization
        self.gradient = gradient
        self.noise = noise
        self.length_coil = length_coil * 1e-3  # We need our length in meters
        self.diameter_coil = diameter_coil * 1e-3  # We need our diameter in meters
        self.number_turns = number_turns
        self.q_factor_transmit = q_factor_transmit
        self.q_factor_receive = q_factor_receive
        self.power_amplifier_power = power_amplifier_power
        self.pulse = pulse
        self.averages = averages
        self.gain = gain
        self.temperature = temperature
        self.loss_TX = loss_TX
        self.loss_RX = loss_RX
        self.conversion_factor = conversion_factor

    def simulate(self):
        reference_voltage = self.calculate_reference_voltage()

        B1 = (
            self.calc_B1() * 1e3
        )  # I think this is multiplied by 1e3 because everything is in mT
        # B1 = 17.3  # Something might be wrong with the calculation of the B1 field. This has to be checked.
        self.sample.gamma = self.sample.gamma * 1e-6  # We need our gamma in MHz / T
        self.sample.T1 = self.sample.T1 * 1e3  # We need our T1 in ms
        self.sample.T2 = self.sample.T2 * 1e3  # We need our T2 in ms

        # Calculate the x distribution of the isochromats
        xdis = self.calc_xdis()

        real_pulsepower = self.pulse.get_real_pulsepower()
        imag_pulsepower = self.pulse.get_imag_pulsepower()

        # Calculate losses on the pulse
        real_pulsepower = real_pulsepower * (1 - 10 ** (-self.loss_TX / 20))
        imag_pulsepower = imag_pulsepower * (1 - 10 ** (-self.loss_TX / 20))

        # Calculate the magnetization
        M_sy1 = self.bloch_symmetric_strang_splitting(
            B1, xdis, real_pulsepower, imag_pulsepower
        )

        # Z-Component
        Mlong = np.squeeze(M_sy1[2, :, :])  # Indices start at 0 in Python
        Mlong_avg = np.mean(Mlong, axis=0)
        Mlong_avg = np.delete(Mlong_avg, -1)  # Remove the last element

        # XY-Component
        Mtrans = np.squeeze(
            M_sy1[1, :, :] + 1j * M_sy1[0, :, :]
        )  # Indices start at 0 in Python
        Mtrans_avg = np.mean(Mtrans, axis=0)
        Mtrans_avg = np.delete(Mtrans_avg, -1)  # Remove the last element

        # Scale the signal according to the reference voltage, averages and gain
        timedomain_signal = Mtrans_avg * reference_voltage

        # Add the losses of the receiver - this should probably be done before the scaling
        timedomain_signal = timedomain_signal * (1 - 10 ** (-self.loss_RX / 20))

        # Add noise to the signal
        noise_data = self.calculate_noise(timedomain_signal)

        timedomain_signal = (timedomain_signal * self.averages * self.gain) + (
            noise_data * self.gain
        )
        # print(abs(timedomain_signal))

        timedomain_signal = timedomain_signal
        return timedomain_signal * self.conversion_factor

    def bloch_symmetric_strang_splitting(
        self, B1, xdis, real_pulsepower, imag_pulsepower, relax=1
    ):
        """This method simulates the Bloch equations using the symmetric strang splitting method.

        Parameters
        ----------
            B1 : float
                The B1 field of the solenoid coil.
            xdis : np.array
                The x distribution of the isochromats.
            real_pulsepower : np.array
                The real part of the pulse power.
            imag_pulsepower : np.array
                The imaginary part of the pulse power.
            relax : float
                If relax = 1, the relaxation is taken into account. If relax = 0, the relaxation is not taken into account.
        """
        Nx = self.number_isochromats
        Nu = real_pulsepower.shape[0]
        M0 = np.array([np.zeros(Nx), np.zeros(Nx), np.ones(Nx)])
        dt = self.pulse.dwell_time * 1e3  # We need our dwell time in ms

        w = np.ones((Nu, 1)) * self.gradient

        # Bloch simulation in magnetization domain
        gadt = self.sample.gamma * dt / 2
        B1 = np.tile(
            (gadt * (real_pulsepower - 1j * imag_pulsepower) * B1).reshape(-1, 1), Nx
        )

        K = gadt * xdis * w * self.gradient
        phi = -np.sqrt(np.abs(B1) ** 2 + K**2)

        cs = np.cos(phi)
        si = np.sin(phi)
        n1 = np.real(B1) / np.abs(phi)
        n2 = np.imag(B1) / np.abs(phi)
        n3 = K / np.abs(phi)
        n1[np.isnan(n1)] = 1
        n2[np.isnan(n2)] = 0
        n3[np.isnan(n3)] = 0
        Bd1 = n1 * n1 * (1 - cs) + cs
        Bd2 = n1 * n2 * (1 - cs) - n3 * si
        Bd3 = n1 * n3 * (1 - cs) + n2 * si
        Bd4 = n2 * n1 * (1 - cs) + n3 * si
        Bd5 = n2 * n2 * (1 - cs) + cs
        Bd6 = n2 * n3 * (1 - cs) - n1 * si
        Bd7 = n3 * n1 * (1 - cs) - n2 * si
        Bd8 = n3 * n2 * (1 - cs) + n1 * si
        Bd9 = n3 * n3 * (1 - cs) + cs

        M = np.zeros((3, Nx, Nu + 1))
        M[:, :, 0] = M0
        Mt = M0
        D = np.diag(
            [
                np.exp(-1 / self.sample.T2 * relax * dt),
                np.exp(-1 / self.sample.T2 * relax * dt),
                np.exp(-1 / self.sample.T1 * relax * dt),
            ]
        )
        b = np.array([0, 0, self.initial_magnetization]) - np.array(
            [
                0,
                0,
                self.initial_magnetization * np.exp(-1 / self.sample.T1 * relax * dt),
            ]
        )

        for n in range(Nu):  # time loop
            Mrot = np.zeros((3, Nx))
            Mrot[0, :] = (
                Bd1.conj().T[:, n] * Mt[0, :]
                + Bd2.conj().T[:, n] * Mt[1, :]
                + Bd3.conj().T[:, n] * Mt[2, :]
            )
            Mrot[1, :] = (
                Bd4.conj().T[:, n] * Mt[0, :]
                + Bd5.conj().T[:, n] * Mt[1, :]
                + Bd6.conj().T[:, n] * Mt[2, :]
            )
            Mrot[2, :] = (
                Bd7.conj().T[:, n] * Mt[0, :]
                + Bd8.conj().T[:, n] * Mt[1, :]
                + Bd9.conj().T[:, n] * Mt[2, :]
            )

            Mt = np.dot(D, Mrot) + np.tile(b, (Nx, 1)).T

            Mrot[0, :] = (
                Bd1.conj().T[:, n] * Mt[0, :]
                + Bd2.conj().T[:, n] * Mt[1, :]
                + Bd3.conj().T[:, n] * Mt[2, :]
            )
            Mrot[1, :] = (
                Bd4.conj().T[:, n] * Mt[0, :]
                + Bd5.conj().T[:, n] * Mt[1, :]
                + Bd6.conj().T[:, n] * Mt[2, :]
            )
            Mrot[2, :] = (
                Bd7.conj().T[:, n] * Mt[0, :]
                + Bd8.conj().T[:, n] * Mt[1, :]
                + Bd9.conj().T[:, n] * Mt[2, :]
            )

            Mt = Mrot
            M[:, :, n + 1] = Mrot

        return M

    def calc_B1(self) -> float:
        """This method calculates the B1 field of our solenoid coil based on the coil parameters and the power amplifier power.

        Returns
        -------
            B1 : float
                The B1 field of the solenoid coil in T."""

        B1 = (
            np.sqrt(2 * self.power_amplifier_power / 50)
            * np.pi
            * np.sqrt(self.q_factor_transmit)
            * 4e-7
            * self.number_turns
            / self.length_coil
        )

        # Spin Factor Scaling 
        B1 = B1 * self.sample.spin_factor

        return B1

    def calc_xdis(self) -> np.array:
        """Calculates the x distribution of the isochromats.

        Returns
        -------
            xdis : np.array
                The x distribution of the isochromats.
        """
        # Df is the Full Width at Half Maximum (FWHM) of Lorentzian in Hz
        Df = 1 / np.pi / self.sample.T2_star

        # Randomly generating frequency offset using Cauchy distribution
        uu = np.random.rand(self.number_isochromats, 1) - 0.5
        foffr = Df / 2 * np.tan(np.pi * uu)

        # xdis is a spatial function, but it is being repurposed here to convert through the gradient to a phase difference per time -> T2 dispersion of the isochromats
        xdis = np.linspace(-1, 1, self.number_isochromats)
        xdis = (
            (foffr.T * 1e-6) / (self.sample.gamma / 2 / np.pi) / (self.gradient * 1e-3)
        )

        return xdis

    def calculate_reference_voltage(self) -> float:
        """This calculates the reference voltage of the measurement setup for the sample at a certain temperature.

        Returns
        -------
            reference_voltage : float
                The reference voltage of the measurement setup for the sample at a certain temperature in Volts.
        """
        u = 4 * np.pi * 1e-7  # permeability of free space

        num, den = self.sample.nuclear_spin.split("/")
        nuclear_spin = float(num) / float(den)

        magnetization = (
            (
                (self.sample.gamma * 2 * self.sample.atoms)
                / (2 * nuclear_spin + 1)
            )
            * (h**2 * self.sample.resonant_frequency)
            / (Boltzmann * self.temperature)
            * self.sample.spin_factor
        )

        coil_crossection = np.pi * (self.diameter_coil / 2) ** 2

        reference_voltage = (
            self.number_turns
            * coil_crossection
            * u
            * (self.sample.resonant_frequency)
            * magnetization
        )
        reference_voltage = (
            reference_voltage * self.sample.powder_factor * self.sample.filling_factor
        )

        # This is assumes that our noise is dominated by everything after the resonator - this is not true for low Q probe coils
        reference_voltage = reference_voltage * np.sqrt(self.q_factor_receive)

        return reference_voltage

    def calculate_noise(self, timedomain_signal: np.array) -> np.array:
        """Calculates the noise array that is added to the signal.

        Parameters
        ----------
            timedomain_signal : np.array
                The time domain signal that is used for the simulation.

        Returns
        -------
            noise_data : np.array
                The noise array that is added to the signal."""
        n_timedomain_points = timedomain_signal.shape[0]
        noise_data = self.noise * 1e-6 * np.random.randn(
            self.averages, n_timedomain_points
        ) + 1j * self.noise * 1e-6 * np.random.randn(self.averages, n_timedomain_points)
        noise_data = np.sum(noise_data, axis=0)  # Sum along the first axis
        return noise_data

    @property
    def sample(self) -> Sample:
        """Sample that is used for the simulation."""
        return self._sample

    @sample.setter
    def sample(self, sample):
        self._sample = sample

    @property
    def number_isochromats(self) -> int:
        """Number of isochromats used for the simulation."""
        return self._number_isochromats

    @number_isochromats.setter
    def number_isochromats(self, number_isochromats):
        self._number_isochromats = number_isochromats

    @property
    def initial_magnetization(self) -> float:
        """Initial magnetization of the sample."""
        return self._initial_magnetization

    @initial_magnetization.setter
    def initial_magnetization(self, initial_magnetization):
        self._initial_magnetization = initial_magnetization

    @property
    def gradient(self) -> float:
        """Gradient of the magnetic field in mt/M."""
        return self._gradient

    @gradient.setter
    def gradient(self, gradient):
        self._gradient = gradient

    @property
    def noise(self) -> float:
        """RMS Noise of the measurement setup in Volts"""
        return self._noise

    @noise.setter
    def noise(self, noise):
        self._noise = noise

    @property
    def length_coil(self) -> float:
        """Length of the coil in meters."""
        return self._length_coil

    @length_coil.setter
    def length_coil(self, length_coil):
        self._length_coil = length_coil

    @property
    def diameter_coil(self) -> float:
        """Diameter of the coil in meters."""
        return self._diameter_coil

    @diameter_coil.setter
    def diameter_coil(self, diameter_coil):
        self._diameter_coil = diameter_coil

    @property
    def number_turns(self) -> float:
        """Number of turns of the coil."""
        return self._number_turns

    @number_turns.setter
    def number_turns(self, number_turns):
        self._number_turns = number_turns

    @property
    def q_factor_transmit(self) -> float:
        """Q-factor of the transmit path of the probe coil."""
        return self._q_factor_transmit

    @q_factor_transmit.setter
    def q_factor_transmit(self, q_factor_transmit):
        self._q_factor_transmit = q_factor_transmit

    @property
    def q_factor_receive(self) -> float:
        """Q-factor of the receive path of the probe coil."""
        return self._q_factor_receive

    @q_factor_receive.setter
    def q_factor_receive(self, q_factor_receive):
        self._q_factor_receive = q_factor_receive

    @property
    def power_amplifier_power(self) -> float:
        """Power of the power amplifier in Watts."""
        return self._power_amplifier_power

    @power_amplifier_power.setter
    def power_amplifier_power(self, power_amplifier_power):
        self._power_amplifier_power = power_amplifier_power

    @property
    def pulse(self) -> PulseArray:
        """Pulse that is used for the simulation."""
        return self._pulse

    @pulse.setter
    def pulse(self, pulse):
        self._pulse = pulse

    @property
    def averages(self) -> int:
        """Number of averages that are used for the simulation."""
        return self._averages

    @averages.setter
    def averages(self, averages):
        self._averages = averages

    @property
    def gain(self) -> float:
        """Gain of the amplifier."""
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @property
    def temperature(self) -> float:
        """Temperature of the sample."""
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature
