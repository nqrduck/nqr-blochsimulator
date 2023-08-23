import numpy as np
import logging
from scipy import signal
from .sample import Sample
from .pulse import PulseArray


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class Simulation:
    """Class for the simulation of the Bloch equations."""
    
    def __init__(
        self,
        sample : Sample,
        number_isochromats : int,
        initial_magnetization : float,
        gradient : float,
        noise : float,
        length_coil : float,
        diameter_coil : float,
        number_turns    : float,
        power_amplifier_power : float,
        pulse : PulseArray
    ):
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
                The RMS Noise of the measurement setup in Volts.
            length_coil : float
                The length of the coil in meters.
            diameter_coil : float
                The diameter of the coil in meters.
            number_turns : float
                The number of turns of the coil.
            power_amplifier_power : float
                The power of the power amplifier in Watts.
            puslse: PulseArray
                The pulse that is used for the simulation.
        """
        self.sample = sample
        self.number_isochromats = number_isochromats
        self.initial_magnetization = initial_magnetization
        self.gradient = gradient
        self.noise = noise
        self.length_coil = length_coil
        self.diameter_coil = diameter_coil
        self.number_turns = number_turns
        self.power_amplifier_power = power_amplifier_power
        self.pulse = pulse

    def simulate(self):
        B1 = self.calc_B1()
        xdis = self.calc_xdis()

        real_pulsepower = self.pulse.get_real_pulsepower()
        imag_pulsepower = self.pulse.get_imag_pulsepower()
        M_sy1 = self.bloch_symmetric_strang_splitting(B1, xdis, real_pulsepower, imag_pulsepower)
        logger.debug("Shape of Msy1: %s", M_sy1.shape)


        # Z-Component
        Mlong = np.squeeze(M_sy1[2,:,:])  # Indices start at 0 in Python
        Mlong_avg = np.mean(Mlong, axis=0)
        Mlong_avg = np.delete(Mlong_avg, -1)  # Remove the last element
        siglong = np.abs(Mlong_avg)

        # XY-Component
        Mtrans = np.squeeze(M_sy1[0,:,:] + 1j*M_sy1[1,:,:])  # Indices start at 0 in Python
        Mtrans_avg = np.mean(Mtrans, axis=0)
        Mtrans_avg = np.delete(Mtrans_avg, -1)  # Remove the last element
        reference = 4.5502
        sigtrans = Mtrans_avg * reference  
        return sigtrans


    def bloch_symmetric_strang_splitting(self, B1, xdis, real_pulsepower, imag_pulsepower, relax = 1):
        """This method simulates the Bloch equations using the symmetric strang splitting method.

        Parameters
        ----------
            B1 : float
                The B1 field of the solenoid coil.
            xdis : np.array
                The x distribution of the isochromats.
        """
        Nx = self.number_isochromats
        Nu = real_pulsepower.shape[0]
        M0 = np.array([np.zeros(Nx), np.zeros(Nx), np.ones(Nx)])
        dt = self.pulse.dwell_time

        w = np.ones((Nu, 1))  * self.gradient

        # Bloch simulation in magnetization domain
        gadt = self.sample.gamma * dt /2 * 1e-3
        B1 = np.tile((gadt * (real_pulsepower - 1j * imag_pulsepower) * B1).reshape(-1, 1), Nx)
        K = gadt * xdis * w * self.gradient
        phi = -np.sqrt(np.abs(B1) ** 2 + K ** 2)

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

        M = np.zeros((3, Nx, Nu+1))
        M[:, :, 0] = M0
        Mt = M0
        D = np.diag([np.exp(-1 / self.sample.T2 * relax * dt), np.exp(-1 / self.sample.T2 * relax * dt), np.exp(-1 / self.sample.T1 * relax * dt)])
        b = np.array([0, 0, self.initial_magnetization]) - np.array([0, 0, self.initial_magnetization * np.exp(-1 / self.sample.T1 * relax * dt)])

        for n in range(Nu): # time loop

            Mrot = np.zeros((3, Nx))
            Mrot[0,:] = Bd1.T[:,n]*Mt[0,:] + Bd2.T[:,n]*Mt[1,:] + Bd3.T[:,n]*Mt[2,:]
            Mrot[1,:] = Bd4.T[:,n]*Mt[0,:] + Bd5.T[:,n]*Mt[1,:] + Bd6.T[:,n]*Mt[2,:]
            Mrot[2,:] = Bd7.T[:,n]*Mt[0,:] + Bd8.T[:,n]*Mt[1,:] + Bd9.T[:,n]*Mt[2,:]

            Mt = np.dot(D, Mrot) + np.tile(b, (Nx, 1)).T

            Mrot[0,:] = Bd1.T[:,n]*Mt[0,:] + Bd2.T[:,n]*Mt[1,:] + Bd3.T[:,n]*Mt[2,:]
            Mrot[1,:] = Bd4.T[:,n]*Mt[0,:] + Bd5.T[:,n]*Mt[1,:] + Bd6.T[:,n]*Mt[2,:]
            Mrot[2,:] = Bd7.T[:,n]*Mt[0,:] + Bd8.T[:,n]*Mt[1,:] + Bd9.T[:,n]*Mt[2,:]

            Mt = Mrot
            M[:, :,n+1] = Mrot

        return M

    def calc_B1(self) -> float:
        """This method calculates the B1 field of our solenoid coil based on the coil parameters and the power amplifier power.
        
        Returns
        -------
            B1 : float
                The B1 field of the solenoid coil."""

        B1 = np.sqrt(2 * self.power_amplifier_power / 50) * np.pi * 4e-7 * self.number_turns / self.length_coil
        return B1
    
    def calc_xdis(self) -> np.array:
        """ Calculates the x distribution of the isochromats. 
        
        Returns
        -------
            xdis : np.array
                The x distribution of the isochromats.
        """
        # Df is the Full Width at Half Maximum (FWHM) of Lorentzian in Hz
        Df = 1 / np.pi / self.sample.T2_star
        logger.debug("Df: %s", Df)

        # Randomly generating frequency offset using Cauchy distribution
        uu = np.random.rand(self.number_isochromats, 1) - 0.5
        foffr = Df / 2 * np.tan(np.pi * uu)

        # xdis is a spatial function, but it is being repurposed here to convert through the gradient to a phase difference per time -> T2 dispersion of the isochromats
        xdis = np.linspace(-1, 1, self.number_isochromats) 
        xdis = (foffr.T) / (self.sample.gamma / (2 * np.pi)) / (self.gradient * 1e-3) 
        return xdis

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
        """ RMS Noise of the measurement setup in Volts"""
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
