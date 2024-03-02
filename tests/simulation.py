import unittest
import numpy as np
import matplotlib.pyplot as plt
from nqr_blochsimulator.classes.sample import Sample
from nqr_blochsimulator.classes.simulation import Simulation
from nqr_blochsimulator.classes.pulse import PulseArray


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.sample = Sample(
            "BiPh3",
            density=1.585e6,  # g/m^3
            molar_mass=440.3,  # g/mol
            resonant_frequency=83.56e6,  # Hz
            gamma=4.342e7,  # Hz/T
            nuclear_spin=9 / 2,
            spin_factor=2,
            powder_factor=0.75,
            filling_factor=0.7,
            T1=83.5e-5,  # s
            T2=396e-6,  # s
            T2_star=50e-6,  # s
        )

        simulation_length = 300e-6
        dwell_time = 1e-6
        self.time_array = np.arange(0, simulation_length, dwell_time)
        pulse_length = 3e-6

        # Simple FID sequence with pulse length of 3µs
        pulse_amplitude_array = np.zeros(int(simulation_length / dwell_time))
        pulse_amplitude_array[: int(pulse_length / dwell_time)] = 1
        pulse_phase_array = np.zeros(int(simulation_length / dwell_time))

        self.pulse = PulseArray(
            pulseamplitude=pulse_amplitude_array,
            pulsephase=pulse_phase_array,
            dwell_time=dwell_time,
        )

        self.simulation = Simulation(
            sample=self.sample,
            number_isochromats=1000,
            initial_magnetization=1,
            gradient=1,
            noise=0.5,
            length_coil=6e-3,
            diameter_coil=3e-3,
            number_turns=9,
            q_factor_transmit=100,
            q_factor_receive=100,
            power_amplifier_power=110,
            pulse=self.pulse,
            averages=1000,
            gain=5600,
            temperature=300,
            loss_TX=12,
            loss_RX=12,
            conversion_factor=2884 # This is for the LimeSDR based spectrometer
        )

    def test_simulation(self):
        M = self.simulation.simulate()

        # Plotting the results
        plt.plot(self.time_array * 1e6, abs(M))
        plt.xlabel("Time (µs)")
        plt.ylabel("Magnetization (a.u.)")
        plt.title("FID of BiPh3")
        plt.show()
