import numpy as np
from numpy import pi
from scipy import signal


class Simulation:
    def __init__(self) -> None:
        pass

    def blochsim(sim_points, sim_time, reference, isochrom, sample, pulse):
        # PRE-SETTINGS
        d = {"M0c": 1}  # initial mag
        NISO = 100
        if isochrom > 0:
            NISO = isochrom  # number of isochromates
        nsamples = (
            sim_points  # number of sample/rasterization points for the calculation
        )
        sim_length = sim_time  # in s; Not larger than the repetition time!
        modulation = "OFF"  # select a optional modulation of the pulse ['OFF','SIN']
        # Replace by the NWA power later
        B1c_calc = np.sqrt(2 * 500 / 50) * pi * 4e-7 * 9 / 6e-3  # 17 od 8.5
        d["B1c"] = 17.3e-3  # for Peak B1 T %12.5%14.3

        # SAMPLE SETTINGS
        d["T1"] = sample["T1"]  # in s; T1, T2
        d["T2"] = sample["T2"]  #
        T2STAR = sample["T2s"]  # only used for some calculations
        d["gamma"] = (
            sample["gamma"] / (2 * pi) / 1e6
        )  # gamma in MHz/T eg 5e6 % sample.gamma in rad/(T s) eg 0.8
        d["relax"] = 1  # Flag 1: with Relax, 0 without Relax

        # Parameter preparation

        # clear up some unit problems
        # DO NOT CHANGE
        d["B1c"] = d["B1c"] * 1e3
        d["T1"] = d["T1"] * 1e3
        d["T2"] = d["T2"] * 1e3
        d["gamma"] = d["gamma"] * 2 * pi

        d["Nx"] = NISO
        d["M0"] = np.array(
            [np.zeros(NISO), np.zeros(NISO), np.ones(NISO)]
        )  # initial magnetization
        d["dt"] = sim_length / nsamples  # time step width
        d["dt"] = (
            d["dt"] * 1e3
        )  # again unit correction. could be changed if necessary, but other time factors
        # in later calculations would have to be changed too

        # Pulse Designer
        u = np.zeros((nsamples, 1))
        v = np.zeros((nsamples, 1))
        w = np.ones((nsamples, 1))

        tt = (np.array(range(1, nsamples + 1)) * d["dt"] - d["dt"]).reshape(
            -1, 1
        )  # time axis in ms

        # PULSE TEMPLATES
        pulse_dur_pow_pha = pulse
        num_pulses, _ = pulse_dur_pow_pha.shape
        # loop through every pulse
        for count in range(num_pulses):
            pulse_begin = pulse_dur_pow_pha[count, 0]
            pulse_end = pulse_dur_pow_pha[count, 1]
            pha = pulse_dur_pow_pha[count, 3] * (2 * pi / 360)  # phase in rad

            ind_begin = np.argmin(np.abs(tt * 1e-3 - pulse_begin))  # minValue is unused
            ind_end = np.argmin(np.abs(tt * 1e-3 - pulse_end))
            ind_end = ind_end - 1

            u_pow, v_pow = np.pol2cart(
                pha, pulse_dur_pow_pha[count, 2]
            )  # theta angle; rho abs

            if modulation == "OFF":
                u[ind_begin:ind_end, 0] = u_pow  # set real pulse power
                v[ind_begin:ind_end, 0] = v_pow  # set imag pulse power
            elif modulation == "SIN":
                u[ind_begin:ind_end, 0] = u_pow * np.sin(
                    (pi * 1e-3 / (pulse_end - pulse_begin)) * tt[ind_begin:ind_end, 0]
                )
                v[ind_begin:ind_end, 0] = v_pow * np.sin(
                    (pi * 1e-3 / (pulse_end - pulse_begin)) * tt[ind_begin:ind_end, 0]
                )

        # Some sidenotes that can be ignored
        for count in range(1):
            d["G3"] = 1  # mT/m, fhwm of 2mm  Gradient scaling
            w = w * d["G3"]  # Gradient in mT/m

            # Isochromatic simulaten by modeling with Lorentz distribution
            Df = 1 / pi / T2STAR  # FWHF of Lorentzian in Hz
            foffr = []
            uu = np.random.rand(NISO, 1) - 0.5
            foffr = Df / 2 * np.tan(pi * uu)  # cauchy distributed frequency offset

            d["xdis"] = np.linspace(-1, 1, NISO)  # in  m spatial resolution
            d["xdis"] = (
                np.array(foffr) * 1e-6 / (d["gamma"] / 2 / pi) / (d["G3"] * 1e-3)
            )  # Conversion factors: foffr from Hz/T to MHz/T as required for d.gamma/2/pi, conversion from Hz-Gamma to radian gamma, and gradient must be scaled from mT/m to T/m

        # USE BLOCH EQUATIONS
        # M_sy1 = bloch_symmetric_strang_splitting_vectorised(u, v, w, d)  # This function would need to be defined or imported

        # Z-Component
        # Mlong = np.squeeze(M_sy1[2, :, :])  # Coordinates M: space components - location(isochromat) - time
        # Mlong_avg = np.mean(Mlong, 1)
        # Mlong_avg = Mlong_avg[:-1]
        # siglong = np.abs(Mlong_avg)

        # XY-Component
        # Mtrans = np.squeeze(M_sy1[0, :, :] + 1j*M_sy1[1, :, :])  # Coordinates M: space components - location(isochromat) - time
        # Mtrans_avg = np.mean(Mtrans, 1)
        # Mtrans_avg = Mtrans_avg[:-1]
        # sigtrans = Mtrans_avg * reference

        # return sigtrans

    def bloch_symmetric_strang_splitting_vectorised(u, v, w, d):
        """Vectorised version of bloch_symmetric_strang_splitting
        
        Parameters
        ----------
        u : array_like
            Real part of pulse
            v : array_like
            Imaginary part of pulse
            w : array_like
            Gradient
            d : dict
        """
        xdis = d["xdis"]
        Nx = d["Nx"]
        Nu = len(u)
        M0 = d["M0"]
        dt = d["dt"]

        gadt = d["gamma"] * dt / 2
        B1 = np.tile(gadt * np.transpose(u - 1j * v) * d["B1c"], (Nx, 1))
        K = gadt * xdis * np.transpose(w) * d["G3"]
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

        M = np.zeros((3, Nx, Nu))
        M[:, :, 0] = M0
        Mt = M0
        D = np.diag(
            [
                np.exp(-1 / d["T2"] * d["relax"] * dt),
                np.exp(-1 / d["T2"] * d["relax"] * dt),
                np.exp(-1 / d["T1"] * d["relax"] * dt),
            ]
        )
        b = np.array([0, 0, d["M0c"]]) - np.array(
            [0, 0, d["M0c"] * np.exp(-1 / d["T1"] * d["relax"] * dt)]
        )

        for n in range(Nu):  # Time loop
            Mrot = np.array(
                [
                    Bd1[:, n] * Mt[0, :] + Bd2[:, n] * Mt[1, :] + Bd3[:, n] * Mt[2, :],
                    Bd4[:, n] * Mt[0, :] + Bd5[:, n] * Mt[1, :] + Bd6[:, n] * Mt[2, :],
                    Bd7[:, n] * Mt[0, :] + Bd8[:, n] * Mt[1, :] + Bd9[:, n] * Mt[2, :],
                ]
            )

            Mt = np.dot(D, Mrot) + np.tile(b, (Nx, 1)).transpose()

            Mrot = np.array(
                [
                    Bd1[:, n] * Mt[0, :] + Bd2[:, n] * Mt[1, :] + Bd3[:, n] * Mt[2, :],
                    Bd4[:, n] * Mt[0, :] + Bd5[:, n] * Mt[1, :] + Bd6[:, n] * Mt[2, :],
                    Bd7[:, n] * Mt[0, :] + Bd8[:, n] * Mt[1, :] + Bd9[:, n] * Mt[2, :],
                ]
            )

            Mt = Mrot
            M[:, :, n + 1] = Mrot

        return M
