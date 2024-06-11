import logging
import numpy as np

logger = logging.getLogger(__name__)

class Sample:
    """
    A class to represent a sample for NQR (Nuclear Quadrupole Resonance) Bloch Simulation.
    """

    avogadro = 6.022e23

    def __init__(
        self,
        name : str,
        atoms : float,
        resonant_frequency : float,
        gamma : float,
        nuclear_spin : str,
        spin_factor : float,
        powder_factor : float,
        filling_factor : float,
        T1 : float,
        T2 : float,
        T2_star : float,
        density : float,
        molar_mass : float,
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
            atoms : float
                The number of atoms per unit volume of the sample (1/m^3).
            resonant_frequency : float
                The resonant frequency of the sample in MHz.
            gamma : float
                The gamma value of the sample in MHz/T.
            nuclear_spin : string
                The nuclear spin quantum number of the sample. Can be half-integer spin.
            spin_factor : float
                The spin transition factor of the sample.
            powder_factor : float
                The powder factor of the sample.
            filling_factor : float
                The filling factor of the sample.
            T1 : float
                The spin-lattice relaxation time of the sample in microseconds.
            T2 : float
                The spin-spin relaxation time of the sample in microseconds.
            T2_star : float
                The effective spin-spin relaxation time of the sample in microseconds.
            density : float
                The density of the sample (g/m^3 or kg/m^3).
            molar_mass : float
                The molar mass of the sample (g/mol or kg/mol).
            atom_density : float, optional
                The atom density of the sample (atoms per cm^3). By default None.
            sample_volume : float, optional
                The volume of the sample (m^3). By default None.
            sample_length : float, optional
                The length of the sample (mm). By default None.
            sample_diameter : float, optional
                The diameter of the sample m(m). By default None.
        """
        self.name = name
        self.atoms = atoms
        self.density = density
        self.molar_mass = molar_mass
        self.resonant_frequency = resonant_frequency * 1e6
        self.gamma = gamma * 1e6
        self.nuclear_spin = nuclear_spin
        self.spin_factor = spin_factor
        self.powder_factor = powder_factor
        self.filling_factor = filling_factor
        self.T1 = T1 * 1e-6
        self.T2 = T2 * 1e-6
        self.T2_star = T2_star * 1e-6
        self.atom_density = atom_density
        self.sample_volume = sample_volume
        self.sample_length = sample_length
        self.sample_diameter = sample_diameter
        if self.atoms == 0:
            self.calculate_atoms()

    # These are helper methods that are used to calculate different parameters of the sample.

    def calculate_atoms(self):
        """
        Calculate the number of atoms in the sample per volume unit (1/m^3). This only works if the sample volume and atom density are provided.
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

        logger.debug(f"Number of atoms in the sample: {self.atoms}")

    def pauli_spin_matrices(self, spin):
        """
        Generate the spin matrices for a given spin value.

        Parameters:
        spin (float): The spin value, which can be a half-integer or integer.

        Returns:
        tuple: A tuple containing the following elements:
            Jx (np.ndarray): The x-component of the spin matrix.
            Jy (np.ndarray): The y-component of the spin matrix.
            Jz (np.ndarray): The z-component of the spin matrix.
            J_minus (np.ndarray): The lowering operator matrix.
            J_plus (np.ndarray): The raising operator matrix.
            m (np.ndarray): The array of magnetic quantum numbers.
        """

        m = np.arange(spin, -spin-1, -1)
        paulirowlength = int(spin * 2 + 1)

        pauli_z = np.diag(m)
        pauli_plus = np.zeros((paulirowlength, paulirowlength))
        pauli_minus = np.zeros((paulirowlength, paulirowlength))

        for row_index in range(paulirowlength - 1):
            col_index = row_index + 1
            pauli_plus[row_index, col_index] = np.sqrt(spin * (spin + 1) - m[col_index] * (m[col_index] + 1))

        for row_index in range(1, paulirowlength):
            col_index = row_index - 1
            pauli_minus[row_index, col_index] = np.sqrt(spin * (spin + 1) - m[col_index] * (m[col_index] - 1))

        Jx = 0.5 * (pauli_plus + pauli_minus)
        Jy = -0.5j * (pauli_plus - pauli_minus)
        Jz = pauli_z

        return Jx, Jy, Jz, pauli_minus, pauli_plus, m
        
    def calculate_spin_transition_factor(self, I, transition):
        """
        Calculate the prefactor for the envisaged spin transition for a given nuclear spin.

        Parameters:
        I (float): The nuclear spin value, which can be a half-integer or integer.
        transition (int): The index of the transition.
                        The transition indices represent the shifts between magnetic quantum numbers m:
                        - 0 represents -1/2 --> 1/2 
                        - 1 represents 1/2 --> 3/2 
                        - 2 represents 3/2 --> 5/2 
                        (only valid transitions based on spin value I are allowed)

        Returns:
        float: The prefactor for the envisaged spin transition.
        """
        m_values = np.arange(I, -I-1, -1)
        if transition < 0 or transition >= len(m_values) - 1:
            raise ValueError(f"Invalid transition for spin {I}. Valid range is 0 to {len(m_values) - 2}")

        Jx, Jy, Jz, J_minus, J_plus, m = self.pauli_spin_matrices(I)
        trindex = int(len(Jx) / 2 - transition)
        spinfactor = Jx[trindex - 1, trindex]

        logger.debug(f"Spin transition factor for I={I}, transition={transition}: {np.real(spinfactor)}")
        logger.info(f"Jx is {Jx}")
        return np.real(spinfactor)
