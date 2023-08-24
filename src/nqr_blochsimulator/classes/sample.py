from math import pi, sqrt


class Sample:
    """
    A class to represent a sample for NQR (Nuclear Quadrupole Resonance) Bloch Simulation.
    """

    avogadro = 6.022e23

    def __init__(
        self,
        name,
        density,
        molar_mass,
        resonant_frequency,
        gamma,
        nuclear_spin,
        spin_factor,
        powder_factor,
        filling_factor,
        T1,
        T2,
        T2_star,
        atom_density=None,
        sample_volume=None,
        sample_length=None,
        sample_diameter=None,
    ):
        """
        Constructs all the necessary attributes for the sample object.

        Parameters
        ----------
            name : str
                The name of the sample.
            density : float
                The density of the sample (g/m^3 or kg/m^3).
            molar_mass : float
                The molar mass of the sample (g/mol or kg/mol).
            resonant_frequency : float
                The resonant frequency of the sample in Hz.
            gamma : float
                The gamma value of the sample in Hz/T.
            nuclear_spin : float
                The nuclear spin quantum number of the sample.
            spin_factor : float
                The spin factor of the sample.
            powder_factor : float
                The powder factor of the sample.
            filling_factor : float
                The filling factor of the sample.
            T1 : float
                The spin-lattice relaxation time of the sample in seconds.
            T2 : float
                The spin-spin relaxation time of the sample in seconds.
            T2_star : float
                The effective spin-spin relaxation time of the sample in seconds.
            atom_density : float, optional
                The atom density of the sample (atoms per cm^3). By default None.
            sample_volume : float, optional
                The volume of the sample (m^3). By default None.
            sample_length : float, optional
                The length of the sample (m). By default None.
            sample_diameter : float, optional
                The diameter of the sample (m). By default None.
        """
        self.name = name
        self.density = density
        self.molar_mass = molar_mass
        self.resonant_frequency = resonant_frequency
        self.gamma = gamma
        self.nuclear_spin = nuclear_spin
        self.spin_factor = spin_factor
        self.powder_factor = powder_factor
        self.filling_factor = filling_factor
        self.T1 = T1
        self.T2 = T2
        self.T2_star = T2_star
        self.atom_density = atom_density
        self.sample_volume = sample_volume
        self.sample_length = sample_length
        self.sample_diameter = sample_diameter
        self.calculate_atoms()

    def calculate_atoms(self):
        """
        Calculate the number of atoms in the sample per volume unit. This only works if the sample volume and atom density are provided.
        Also the sample should be cylindrical.

        If atom density and sample volume are provided, use these to calculate the number of atoms.
        If not, use Avogadro's number, density, and molar mass to calculate the number of atoms.
        """
        if self.atom_density and self.sample_volume:
            self.atoms = (
                self.atom_density
                * self.sample_volume
                / 1e-6
                / (self.sample_volume * self.sample_length / self.sample_diameter)
            )
        else:
            self.atoms = self.avogadro * self.density / self.molar_mass
