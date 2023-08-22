import unittest
from nqr_blochsimulator.classes.sample import Sample
from nqr_blochsimulator.classes.simulation import Simulation

class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.sample = Sample(
            "Ammonium nitrate",
            1720,
            80.0433
            * 1e-3
            / Simulation.avogadro,  # molar mass in kg/mol
            1.945e6,
            2 * 3.436e8,
            1.5,
            0.5,
            1,
            0.1,
            0.1,
            0.1,
            0.1,
        )
        self.simulation = Simulation(self.sample, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6)

        