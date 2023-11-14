# -*- coding: utf-8 -*-
"""Supernova oscillation physics for flavors e, X, e-bar, X-bar.

For measured mixing angles and latest global analysis results, visit
http://www.nu-fit.org/.
"""

from abc import abstractmethod, ABC

import numpy as np
from astropy import units as u
from astropy import constants as c

from .neutrino import MassHierarchy, MixingParameters


class FlavorTransformation(ABC):
    """Generic interface to compute neutrino and antineutrino survival probability."""

    @abstractmethod
    def prob_ee(self, t, E):
        """Electron neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        float or ndarray
            Transition probability.
        """
        pass

    @abstractmethod
    def prob_ex(self, t, E):
        """X -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        float or ndarray
            Transition probability.
        """
        pass

    @abstractmethod
    def prob_xx(self, t, E):
        """Flavor X neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        float or ndarray
            Transition probability.
        """
        pass

    @abstractmethod
    def prob_xe(self, t, E):
        """e -> X neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        float or ndarray
            Transition probability.
        """
        pass

    @abstractmethod
    def prob_eebar(self, t, E):
        """Electron antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        float or ndarray
            Transition probability.
        """
        pass

    @abstractmethod
    def prob_exbar(self, t, E):
        """X -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        float or ndarray
            Transition probability.
        """
        pass

    @abstractmethod
    def prob_xxbar(self, t, E):
        """X -> X antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        float or ndarray
            Transition probability.
        """
        pass

    @abstractmethod
    def prob_xebar(self, t, E):
        """e -> X antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        float or ndarray
            Transition probability.
        """
        pass


class NoTransformation(FlavorTransformation):
    """Survival probabilities for no oscillation case."""

    def __init__(self):
        pass

    def prob_ee(self, t, E):
        """Electron neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1.

    def prob_ex(self, t, E):
        """X -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_ee(t,E)

    def prob_xx(self, t, E):
        """Flavor X neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_ee(t,E)) / 2.

    def prob_xe(self, t, E):
        """e -> X neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_ee(t,E)) / 2.

    def prob_eebar(self, t, E):
        """Electron antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1.

    def prob_exbar(self, t, E):
        """X -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_eebar(t,E)

    def prob_xxbar(self, t, E):
        """X -> X antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_eebar(t,E)) / 2.

    def prob_xebar(self, t, E):
        """e -> X antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_eebar(t,E)) / 2.


class CompleteExchange(FlavorTransformation):
    """Survival probabilities for the case when the electron flavors are completely exchanged with the x flavor."""

    def __init__(self):
        pass

    def prob_ee(self, t, E):
        """Electron neutrino survival probability.
        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.
        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 0.

    def prob_ex(self, t, E):
        """X -> e neutrino transition probability.
        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.
        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_ee(t,E)

    def prob_xx(self, t, E):
        """Flavor X neutrino survival probability.
        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.
        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_ee(t,E)) / 2.

    def prob_xe(self, t, E):
        """e -> X neutrino transition probability.
        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.
        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_ee(t,E)) / 2.

    def prob_eebar(self, t, E):
        """Electron antineutrino survival probability.
        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.
        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 0.

    def prob_exbar(self, t, E):
        """X -> e antineutrino transition probability.
        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.
        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_eebar(t,E)

    def prob_xxbar(self, t, E):
        """X -> X antineutrino survival probability.
        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.
        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_eebar(t,E)) / 2.

    def prob_xebar(self, t, E):
        """e -> X antineutrino transition probability.
        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.
        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_eebar(t,E)) / 2.


class AdiabaticMSW(FlavorTransformation):
    """Adiabatic MSW effect."""

    def __init__(self, mix_angles=None, mh=MassHierarchy.NORMAL):
        """Initialize transformation matrix.

        Parameters
        ----------
        mix_angles : tuple or None
            If not None, override default mixing angles using tuple (theta12, theta13, theta23).
        mh : MassHierarchy
            MassHierarchy.NORMAL or MassHierarchy.INVERTED.
        """
        if type(mh) == MassHierarchy:
            self.mass_order = mh
        else:
            raise TypeError('mh must be of type MassHierarchy')

        if mix_angles is not None:
            theta12, theta13, theta23 = mix_angles
        else:
            pars = MixingParameters(mh)
            theta12, theta13, theta23 = pars.get_mixing_angles()

        self.De1 = float((np.cos(theta12) * np.cos(theta13))**2)
        self.De2 = float((np.sin(theta12) * np.cos(theta13))**2)
        self.De3 = float(np.sin(theta13)**2)

    def prob_ee(self, t, E):
        """Electron neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De3
        else:
            return self.De2

    def prob_ex(self, t, E):
        """X -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_ee(t,E)

    def prob_xx(self, t, E):
        """Flavor X neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_ee(t,E)) / 2.

    def prob_xe(self, t, E):
        """e -> X neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_ee(t,E)) / 2.

    def prob_eebar(self, t, E):
        """Electron antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De1
        else:
            return self.De3

    def prob_exbar(self, t, E):
        """X -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_eebar(t,E)

    def prob_xxbar(self, t, E):
        """X -> X antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_eebar(t,E)) / 2.

    def prob_xebar(self, t, E):
        """e -> X antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_eebar(t,E)) / 2.


class AdiabaticMSWProject(FlavorTransformation):
    """Adiabatic MSW effect."""

    """ We are defining a new class for the paper to be use. Hence we don't alter the existing class"""

    def __init__(self, mix_angles=None, mh=MassHierarchy.NORMAL):
        """Initialize transformation matrix.

        Parameters
        ----------
        mix_angles : tuple or None
            If not None, override default mixing angles using tuple (theta12, theta13, theta23).
        mh : MassHierarchy
            MassHierarchy.NORMAL or MassHierarchy.INVERTED.
        """
        if type(mh) == MassHierarchy:
            self.mass_order = mh
        else:
            raise TypeError('mh must be of type MassHierarchy')

        if mix_angles is not None:
            theta12, theta13, theta23 = mix_angles
            deltaCP = 197*u.deg  # Setting this as default if mixing angles are provided. To adjust later. For now
            # just to avoid crashing
        else:
            pars = MixingParameters(mh)
            theta12, theta13, theta23 = pars.get_mixing_angles()
            deltaCP = pars.get_deltaCP(self)

        deltaCP_rad = deltaCP.to(u.rad).value # Converting to radian. Not neccessary for angles.
        e_mdeltaCP = np.exp(-1j*deltaCP_rad)
        e_deltaCP = np.exp(1j*deltaCP_rad)

        # A few precomputation.

        c_12=float(np.cos(theta12))
        s_12=float(np.sin(theta12))
        c_13=float(np.cos(theta13))
        s_13=float(np.sin(theta13))
        c_23=float(np.cos(theta23))
        s_23=float(np.sin(theta23))
        ## Computing PMNX elements

        # \nu_e row
        self.U_e1=c_13*c_12
        self.U_e2=c_13*s_12
        self.U_e3=s_13*s_12*e_mdeltaCP

        # \nu_\mu row
        self.U_u1 = -c_23*s_12-s_23*s_13*c_12*e_deltaCP
        self.U_u2 = c_23*c_12 -s_23*s_13*s_12*e_deltaCP
        self.U_u3=  s_23*c_13

        # \nu_\tau row
        self.U_t1= s_23*s_12-c_23*s_13*c_12*e_deltaCP
        self.U_t2=-s_23*c_12-c_23*s_13*s_12*e_deltaCP
        self.U_t3= c_23*c_13

    def prob_ee(self, t, E):
        """Electron neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return abs(self.U_e3)**2
        else:
            return abs(self.U_e2)**2

    def prob_ex(self, t, E):
        """X -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_ee(t,E)

    def prob_xx(self, t, E):
        """Flavor X neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """

        if self.mass_order == MassHierarchy.NORMAL:
            return (abs(self.U_u1)**2+abs(self.U_t1)**2+abs(self.U_u2)**2+abs(self.U_t2)**2) / 2.
        else:
            return (abs(self.U_u1)**2+abs(self.U_t1)**2+abs(self.U_u3)**2+abs(self.U_t3)**2) / 2.



    def prob_xe(self, t, E):
        """e -> X neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """

         if self.mass_order == MassHierarchy.NORMAL:
            return (abs(self.U_u3)**2+abs(self.U_t3)**2) / 2.
        else:
            return (abs(self.U_u2)**2+abs(self.U_t2)**2) / 2.


    def prob_eebar(self, t, E):
        """Electron antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return abs(self.U_e1)**2
        else:
            return abs(self.U_e3)**2

    def prob_exbar(self, t, E):
        """X -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_eebar(t,E)

    def prob_xxbar(self, t, E):
        """X -> X antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return (abs(self.U_u2)**2+abs(self.U_t2)**2+abs(self.U_u3)**2+abs(self.U_t3)**2) / 2.
        else:
            return (abs(self.U_u1)**2+abs(self.U_t1)**2+abs(self.U_u2)**2+abs(self.U_t2)**2) / 2.


    def prob_xebar(self, t, E):
        """e -> X antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
         if self.mass_order == MassHierarchy.NORMAL:
            return (abs(self.U_u1)**2+abs(self.U_t1)**2)/ 2.
        else:
            return (abs(self.U_u3)**2+abs(self.U_t3)**2) / 2.



class NonAdiabaticMSWH(FlavorTransformation):
    """Nonadiabatic MSW effect."""

    def __init__(self, mix_angles=None, mh=MassHierarchy.NORMAL):
        """Initialize transformation matrix.

        Parameters
        ----------
        mix_angles : tuple or None
            If not None, override default mixing angles using tuple (theta12, theta13, theta23).
        mh : MassHierarchy
            MassHierarchy.NORMAL or MassHierarchy.INVERTED.
        """
        if type(mh) == MassHierarchy:
            self.mass_order = mh
        else:
            raise TypeError('mh must be of type MassHierarchy')

        if mix_angles is not None:
            theta12, theta13, theta23 = mix_angles
        else:
            pars = MixingParameters(mh)
            theta12, theta13, theta23 = pars.get_mixing_angles()

        self.De1 = float((np.cos(theta12) * np.cos(theta13))**2)
        self.De2 = float((np.sin(theta12) * np.cos(theta13))**2)
        self.De3 = float(np.sin(theta13)**2)

    def prob_ee(self, t, E):
        """Electron neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return self.De2

    def prob_ex(self, t, E):
        """X -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_ee(t,E)

    def prob_xx(self, t, E):
        """Flavor X neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_ee(t,E)) / 2.

    def prob_xe(self, t, E):
        """e -> X neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_ee(t,E)) / 2.

    def prob_eebar(self, t, E):
        """Electron antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return self.De1

    def prob_exbar(self, t, E):
        """X -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_eebar(t,E)

    def prob_xxbar(self, t, E):
        """X -> X antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_eebar(t,E)) / 2.

    def prob_xebar(self, t, E):
        """e -> X antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_eebar(t,E)) / 2.


class TwoFlavorDecoherence(FlavorTransformation):
    """Star-earth transit survival probability: two flavor case."""

    def __init__(self, mix_angles=None, mh=MassHierarchy.NORMAL):
        """Initialize transformation matrix.

        Parameters
        ----------
        mix_angles : tuple or None
            If not None, override default mixing angles using tuple (theta12, theta13, theta23).
        mh : MassHierarchy
            MassHierarchy.NORMAL or MassHierarchy.INVERTED.
        """
        if type(mh) == MassHierarchy:
            self.mass_order = mh
        else:
            raise TypeError('mh must be of type MassHierarchy')

        if mix_angles is not None:
            theta12, theta13, theta23 = mix_angles
        else:
            pars = MixingParameters(mh)
            theta12, theta13, theta23 = pars.get_mixing_angles()

        self.De1 = float((np.cos(theta12) * np.cos(theta13))**2)
        self.De2 = float((np.sin(theta12) * np.cos(theta13))**2)
        self.De3 = float(np.sin(theta13)**2)

    def prob_ee(self, t, E):
        """Electron neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return (self.De2+self.De3)/2.
        else:
            return self.De2

    def prob_ex(self, t, E):
        """X -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_ee(t,E)

    def prob_xx(self, t, E):
        """Flavor X neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_ee(t,E)) / 2.

    def prob_xe(self, t, E):
        """e -> X neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_ee(t,E)) / 2.

    def prob_eebar(self, t, E):
        """Electron antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De1
        else:
            return (self.De1+self.De3)/2

    def prob_exbar(self, t, E):
        """X -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_eebar(t,E)

    def prob_xxbar(self, t, E):
        """X -> X antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_eebar(t,E)) / 2.

    def prob_xebar(self, t, E):
        """e -> X antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_eebar(t,E)) / 2.


class ThreeFlavorDecoherence(FlavorTransformation):
    """Star-earth transit survival probability: three flavor case."""

    def __init__(self):
        pass

    def prob_ee(self, t, E):
        """Electron neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.
        """
        return 1./3.

    def prob_ex(self, t, E):
        """X -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_ee(t,E)

    def prob_xx(self, t, E):
        """Flavor X neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_ee(t,E)) / 2.

    def prob_xe(self, t, E):
        """e -> X neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_ee(t,E)) / 2.

    def prob_eebar(self, t, E):
        """Electron antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1./3.

    def prob_exbar(self, t, E):
        """X -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_eebar(t,E)

    def prob_xxbar(self, t, E):
        """X -> X antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. + self.prob_eebar(t,E)) / 2.

    def prob_xebar(self, t, E):
        """e -> X antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_eebar(t,E)) / 2.


class NeutrinoDecay(FlavorTransformation):
    """Decay effect, where the heaviest neutrino decays to the lightest
    neutrino. For a description and typical parameters, see A. de Gouvêa et al.,
    PRD 101:043013, 2020, arXiv:1910.01127.
    """
    def __init__(self, mix_angles=None, mass=1*u.eV/c.c**2, tau=1*u.day, dist=10*u.kpc, mh=MassHierarchy.NORMAL):
        """Initialize transformation matrix.

        Parameters
        ----------
        mix_angles : tuple or None
            If not None, override default mixing angles using tuple (theta12, theta13, theta23).
        mass : astropy.units.quantity.Quantity
            Mass of the heaviest neutrino; expect in eV/c^2.
        tau : astropy.units.quantity.Quantity
            Lifetime of the heaviest neutrino.
        dist : astropy.units.quantity.Quantity
            Distance to the supernova.
        mh : MassHierarchy
            MassHierarchy.NORMAL or MassHierarchy.INVERTED.
        """
        if type(mh) == MassHierarchy:
            self.mass_order = mh
        else:
            raise TypeError('mh must be of type MassHierarchy')

        if mix_angles is not None:
            theta12, theta13, theta23 = mix_angles
        else:
            pars = MixingParameters(mh)
            theta12, theta13, theta23 = pars.get_mixing_angles()

        self.De1 = float((np.cos(theta12) * np.cos(theta13))**2)
        self.De2 = float((np.sin(theta12) * np.cos(theta13))**2)
        self.De3 = float(np.sin(theta13)**2)

        self.m = mass
        self.tau = tau
        self.d = dist

    def gamma(self, E):
        """Decay width of the heaviest neutrino mass.

        Parameters
        ----------
        E : float
            Energy of the nu3.

        Returns
        -------
        Gamma : float
            Decay width of the neutrino mass, in units of 1/length.

        :meta private:
        """
        return self.m*c.c / (E*self.tau)

    def prob_ee(self, t, E):
        """Electron neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        # NMO case.
        if self.mass_order == MassHierarchy.NORMAL:
            pe_array = self.De1*(1-np.exp(-self.gamma(E)*self.d)) + \
                       self.De3*np.exp(-self.gamma(E)*self.d)
        # IMO case.
        else:
            pe_array = self.De2*np.exp(-self.gamma(E)*self.d) + \
                       self.De3*(1-np.exp(-self.gamma(E)*self.d))
        return pe_array

    def prob_ex(self, t, E):
        """X -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        # NMO case.
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De1 + self.De3
        # IMO case.
        else:
            return self.De1 + self.De2

    def prob_xx(self, t, E):
        """Flavor X neutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_ex(t,E) / 2.

    def prob_xe(self, t, E):
        """e -> X neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_ee(t,E)) / 2.

    def prob_eebar(self, t, E):
        """Electron antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return self.De3

    def prob_exbar(self, t, E):
        """X -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        # NMO case.
        if self.mass_order == MassHierarchy.NORMAL:
            pxbar_array = self.De1*(1-np.exp(-self.gamma(E)*self.d)) + \
                          self.De2 + self.De3*np.exp(-self.gamma(E)*self.d)
        # IMO case.
        else:
            pxbar_array = self.De1 + self.De2*np.exp(-self.gamma(E)*self.d) + \
                          self.De3*(1-np.exp(-self.gamma(E)*self.d))
        return pxbar_array

    def prob_xxbar(self, t, E):
        """X -> X antineutrino survival probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return 1. - self.prob_exbar(t,E) / 2.

    def prob_xebar(self, t, E):
        """e -> X antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return (1. - self.prob_eebar(t,E)) / 2.


class AdiabaticMSWes(FlavorTransformation):
    """A four-neutrino mixing prescription. The assumptions used are that:

    1. the fourth neutrino mass is the heaviest but not so large that the electron-sterile resonances
       are inside the neutrinosphere;
    2. the “outer” or H' electron-sterile MSW resonance is adiabatic;
    3. the “inner” or H'' electron-sterile MSW resonance (where the electron fraction = 1/3) is non-adiabatic.

    For further insight see, for example, Esmaili, Peres, and Serpico, Phys. Rev. D 90, 033013 (2014).
    """
    def __init__(self, mix_angles, mh=MassHierarchy.NORMAL):
        """Initialize transformation matrix.

        Parameters
        ----------
        mix_angles : tuple
            Values for mixing angles (theta12, theta13, theta23, theta14).
        mh : MassHierarchy
            MassHierarchy.NORMAL or MassHierarchy.INVERTED.
        """
        if type(mh) == MassHierarchy:
            self.mass_order = mh
        else:
            raise TypeError('mh must be of type MassHierarchy')

        theta12, theta13, theta23, theta14 = mix_angles

        self.De1 = float((np.cos(theta12) * np.cos(theta13) * np.cos(theta14))**2)
        self.De2 = float((np.sin(theta12) * np.cos(theta13) * np.cos(theta14))**2)
        self.De3 = float((np.sin(theta13) * np.cos(theta14))**2)
        self.De4 = float((np.sin(theta14))**2)
        self.Ds1 = float((np.cos(theta12) * np.cos(theta13) * np.sin(theta14))**2)
        self.Ds2 = float((np.sin(theta12) * np.cos(theta13) * np.sin(theta14))**2)
        self.Ds3 = float((np.sin(theta13) * np.sin(theta14))**2)
        self.Ds4 = float((np.cos(theta14))**2)

    def prob_ee(self, t, E):
        """e -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return self.De4

    def prob_ex(self, t, E):
        """x -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De1 + self.De2
        else:
            return self.De1 + self.De3

    def prob_xx(self, t, E):
        """x -> x neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return ( 2 - self.De1 - self.De2 - self.Ds1 - self.Ds2 ) / 2
        else:
            return ( 2 - self.De1 - self.De3 - self.Ds1 - self.Ds3 ) / 2

    def prob_xe(self, t, E):
        """e -> x neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        return ( 1 - self.De4 - self.Ds4 )/2

    def prob_eebar(self, t, E):
        """e -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De1
        else:
            return self.De3

    def prob_exbar(self, t, E):
        """x -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De3 + self.De4
        else:
            return self.De2 + self.De4

    def prob_xxbar(self, t, E):
        """x -> x antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return ( 2 - self.De3 - self.De4 - self.Ds3 - self.Ds4 ) / 2
        else:
            return ( 2 - self.De2 - self.De4 - self.Ds2 - self.Ds4 ) / 2

    def prob_xebar(self, t, E):
        """e -> x antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return ( 1 - self.De1 - self.Ds1 ) / 2
        else:
            return ( 1 - self.De3 - self.Ds3 ) / 2


class NonAdiabaticMSWes(FlavorTransformation):
    """A four-neutrino mixing prescription. The assumptions used are that:

    1. the fourth neutrino mass is the heaviest but not so large that the electron-sterile resonances
       are inside the neutrinosphere;
    2. the “outer” or H' electron-sterile MSW resonance is non-adiabatic;
    3. the “inner” or H'' electron-sterile MSW resonance (where the electron fraction = 1/3) is non-adiabatic.

    For further insight see, for example, Esmaili, Peres, and Serpico, Phys. Rev. D 90, 033013 (2014).
    """
    def __init__(self, mix_angles, mh=MassHierarchy.NORMAL):
        """Initialize transformation matrix.

        Parameters
        ----------
        mix_angles : tuple
            Values for mixing angles (theta12, theta13, theta23, theta14).
        mh : MassHierarchy
            MassHierarchy.NORMAL or MassHierarchy.INVERTED.
        """
        if type(mh) == MassHierarchy:
            self.mass_order = mh
        else:
            raise TypeError('mh must be of type MassHierarchy')

        theta12, theta13, theta23, theta14 = mix_angles

        self.De1 = float((np.cos(theta12) * np.cos(theta13) * np.cos(theta14))**2)
        self.De2 = float((np.sin(theta12) * np.cos(theta13) * np.cos(theta14))**2)
        self.De3 = float((np.sin(theta13) * np.cos(theta14))**2)
        self.De4 = float((np.sin(theta14))**2)
        self.Ds1 = float((np.cos(theta12) * np.cos(theta13) * np.sin(theta14))**2)
        self.Ds2 = float((np.sin(theta12) * np.cos(theta13) * np.sin(theta14))**2)
        self.Ds3 = float((np.sin(theta13) * np.sin(theta14))**2)
        self.Ds4 = float((np.cos(theta14))**2)

    def prob_ee(self, t, E):
        """e -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De3
        else:
            return self.De2

    def prob_ex(self, t, E):
        """x -> e neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De1 + self.De2
        else:
            return self.De1 + self.De3

    def prob_xx(self, t, E):
        """x -> x neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return ( 2 - self.De1 - self.De2 - self.Ds1 - self.Ds2 ) / 2
        else:
            return ( 2 - self.De1 - self.De3 - self.Ds1 - self.Ds3 ) / 2

    def prob_xe(self, t, E):
        """e -> x neutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return (1 - self.De3 - self.Ds3)/2
        else:
            return (1 - self.De2 - self.Ds2) / 2

    def prob_eebar(self, t, E):
        """e -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De1
        else:
            return self.De3

    def prob_exbar(self, t, E):
        """x -> e antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return self.De2 + self.De3
        else:
            return self.De1 + self.De2

    def prob_xxbar(self, t, E):
        """x -> x antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return ( 2 - self.De2 - self.De3 - self.Ds2 - self.Ds3 ) / 2
        else:
            return ( 2 - self.De1 - self.De2 - self.Ds1 - self.Ds2 ) / 2

    def prob_xebar(self, t, E):
        """e -> x antineutrino transition probability.

        Parameters
        ----------
        t : float or ndarray
            List of times.
        E : float or ndarray
            List of energies.

        Returns
        -------
        prob : float or ndarray
            Transition probability.
        """
        if self.mass_order == MassHierarchy.NORMAL:
            return ( 1 - self.De1 - self.Ds1 ) / 2
        else:
            return ( 1 - self.De3 - self.Ds3 ) / 2
