import itertools as it
import os
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
from astropy import units as u
from scipy.stats import norm

from astropy.table import Table, join
from astropy.units import UnitTypeError, get_physical_type
from astropy.units.quantity import Quantity
from astropy import constants as aconst
from scipy.special import loggamma
from snewpy import _model_downloader

from snewpy.neutrino import Flavor
from snewpy.neutrino import PreComputingU_PMNSElementSquared
from snewpy.flavor_transformation import NoTransformation
from functools import wraps
from snewpy.flux import Flux

PrecomputingU_PMNSelementsquared=PreComputingU_PMNSElementSquared()

def _wrap_init(init, check):
    @wraps(init)
    def _wrapper(self, *arg, **kwargs):
        init(self, *arg, **kwargs)
        check(self)
    return _wrapper


class SupernovaModel(ABC):
    """Base class defining an interface to a supernova model."""

    def __init_subclass__(cls, **kwargs):
        """Hook to modify the subclasses on creation"""
        super().__init_subclass__(**kwargs)
        cls.__init__ = _wrap_init(cls.__init__, cls.__post_init_check)

    def __init__(self, time, QCD_effect_time, BH_effect_time, metadata):
        """Initialize supernova model base class
        (call this method in the subclass constructor as ``super().__init__(time,metadata)``).

        Parameters
        ----------
        time : ndarray of astropy.Quantity
            Time points where the model flux is defined.
            Must be array of :class:`Quantity`, with units convertable to "second".
        metadata : dict
            Dict of model parameters <name>:<value>,
            to be used for printing table in :meth:`__repr__` and :meth:`_repr_markdown_`
        """
        self.time = time
        self.metadata = metadata

    def __repr__(self):
        """Default representation of the model.
        """

        mod = f"{self.__class__.__name__} Model"
        try:
            mod += f': {self.filename}'
        except AttributeError:
            pass
        s = [mod]
        for name, v in self.metadata.items():
            s += [f"{name:16} : {v}"]
        return '\n'.join(s)

    def __post_init_check(self):
        """A function to check model integrity after initialization"""
        try:
            t = self.time
            m = self.metadata
        except AttributeError as e:
            clsname = self.__class__.__name__
            raise TypeError(f"Model not initialized. Please call 'SupernovaModel.__init__' within the '{clsname}.__init__'") from e

    def _repr_markdown_(self):
        """Markdown representation of the model, for Jupyter notebooks.
        """
        mod = f'**{self.__class__.__name__} Model**'
        try:
            mod +=f': {self.filename}'
        except:
            pass
        s = [mod,'']
        if self.metadata:
            s += ['|Parameter|Value|',
                  '|:--------|:----:|']
            for name, v in self.metadata.items():
                try:
                    s += [f"|{name} | ${v.value:g}$ {v.unit:latex}|"]
                except:
                    s += [f"|{name} | {v} |"]
        return '\n'.join(s)

    def get_time(self):
        """Returns
        -------
        ndarray of astropy.Quantity
            Snapshot times from the simulation
        """
        return self.time

    @abstractmethod
    def get_initial_spectra(self, t, E, flavors=Flavor):
        """Get neutrino spectra at the source.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial spectra.
        flavors: iterable of snewpy.neutrino.Flavor
            Return spectra for these flavors only (default: all)

        Returns
        -------
        initialspectra : dict
            Dictionary of neutrino spectra, keyed by neutrino flavor.
        """
        pass

    def get_initialspectra(self, *args):
        """DO NOT USE! Only for backward compatibility!

        :meta private:
        """
        warn("Please use `get_initial_spectra()` instead of `get_initialspectra()`!", FutureWarning)
        return self.get_initial_spectra(*args)

    def get_transformed_spectra(self, t, E, flavor_xform):
        """Get neutrino spectra after applying oscillation.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial and oscillated spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial and oscillated spectra.
        flavor_xform : FlavorTransformation
            An instance from the flavor_transformation module.

        Returns
        -------
        dict
            Dictionary of transformed spectra, keyed by neutrino flavor.
        """
        initialspectra = self.get_initial_spectra(t, E)
        transformed_spectra = {}

        transformed_spectra[Flavor.NU_E] = \
            flavor_xform.prob_ee(t, E) * initialspectra[Flavor.NU_E] + \
            flavor_xform.prob_ex(t, E) * initialspectra[Flavor.NU_X]

        transformed_spectra[Flavor.NU_X] = \
            flavor_xform.prob_xe(t, E) * initialspectra[Flavor.NU_E] + \
            flavor_xform.prob_xx(t, E) * initialspectra[Flavor.NU_X]

        transformed_spectra[Flavor.NU_E_BAR] = \
            flavor_xform.prob_eebar(t, E) * initialspectra[Flavor.NU_E_BAR] + \
            flavor_xform.prob_exbar(t, E) * initialspectra[Flavor.NU_X_BAR]

        transformed_spectra[Flavor.NU_X_BAR] = \
            flavor_xform.prob_xebar(t, E) * initialspectra[Flavor.NU_E_BAR] + \
            flavor_xform.prob_xxbar(t, E) * initialspectra[Flavor.NU_X_BAR]


        #print(transformed_spectra[Flavor.NU_E])
        #print(transformed_spectra[Flavor.NU_E_BAR])

        return transformed_spectra

    def get_flux (self, t, E, distance, flavor_xform=NoTransformation()):
        """Get neutrino flux through 1cm^2 surface at the given distance

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate the neutrino spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the the neutrino spectra.
        distance : astropy.Quantity or float (in kpc)
            Distance from supernova.
        flavor_xform : FlavorTransformation
            An instance from the flavor_transformation module.

        Returns
        -------
        dict
            Dictionary of neutrino fluxes in [neutrinos/(cm^2*erg*s)],
            keyed by neutrino flavor.

        """
        distance = distance << u.kpc #assume that provided distance is in kpc, or convert
        factor = 1/(4*np.pi*(distance.to('cm'))**2)
        f = self.get_transformed_spectra(t, E, flavor_xform)

        array = np.stack([f[flv] for flv in sorted(Flavor)])
        return  Flux(data=array*factor, flavor=np.sort(Flavor), time=t, energy=E)



    def get_transformed_spectra_project_arbitrary_masses(self, t, E, distance, neutrino_masses):
            """Get neutrino spectra after applying oscillation.

            This method handle arbitrary masses ordering.

            See paper or Appendix of paper [insert ArXiv link]

            \Phi_{\nu_{e}}(E, t)       =\left|U_{e H}\right|^{2}\Phi_{\nu_{e}}^{0}(E, t) + \left(\left|U_{e L}\right|^{2}+\left|U_{e M}\right|^{2}\right) \Phi_{\nu_{x}}^{0}(E, t)
            \Phi_{\bar{\nu}_{e}}(E, t) =\left|U_{e L}\right|^{2}\Phi_{\bar{\nu}_{e}}^{0}(E, t) + \left(\left|U_{e M}\right|^{2}+\left|U_{e H}\right|^{2}\right) \Phi_{\bar{\nu}_{x}}^{0}(E, t)
            2\Phi_{\nu_{x}}(E, t)      =\left(\left|U_{\mu H}\right|^{2}+\left|U_{\tau H}\right|^{2}\right)\Phi_{\nu_{e}}^{0}(E, t)+\left(\left|U_{\mu L}\right|^{2}+\left|U_{\tau L}\right|^{2}+\left|U_{\mu M}\right|^{2}+\left|U_{\tau M}\right|^{2}\right) \Phi_{\nu_{x}}^{0}(E, t)
            2\Phi_{\bar{\nu}_{x}}(E, t)=\left(\left|U_{\mu L}\right|^{2}+\left|U_{\tau L}\right|^{2}\right)\Phi_{\bar{\nu}_{e}}^{0}(E, t)+\left(\left|U_{\mu M}\right|^{2}+\left|U_{\tau M}\right|^{2}+\left|U_{\mu H}\right|^{2}+\left|U_{\tau H}\right|^{2}\right) \Phi_{\bar{\nu}_{x}}^{0}(E, t)


            Parameters
            ----------
            t : astropy.Quantity
                Time to evaluate initial and oscillated spectra.
            E : astropy.Quantity or ndarray of astropy.Quantity
                Energies to evaluate the initial and oscillated spectra.

            distance : float
                distance to the supernova assumed to be in kpc

            neutrino_masses: None or List
                Neutrino masses list [m1,m2,m3]. If None, then used the default SNEWPY
            mass_hierachy: str
                mass_hierachy="NO" for Normal ordering or "IO" for INVERTED ordering
            snmodel_dict : dict

            Returns
            -------
            dict
                Dictionary of transformed spectra, keyed by neutrino flavor.
            """



            ####################### Computing time delay effect first ###################

            distance = distance << u.kpc   # Assume distance is in Kpc
            # Computing time delay dela_ts
            neutrino_masses = neutrino_masses  << u.eV   # Assume the masses are given in eV
              # Converting to MeV

            # Compute initial flux with time delays
            shifts={}
            shifts[0]=self.get_initial_spectra_arbitrary(t, E, neutrino_masses[0], distance)
            shifts[1]=self.get_initial_spectra_arbitrary(t, E, neutrino_masses[1], distance)
            shifts[2]=self.get_initial_spectra_arbitrary(t, E, neutrino_masses[2], distance)


            U=PrecomputingU_PMNSelementsquared
            # Add plus one for in the indices to match U_PMNS notation
            sorted_indices =np.argsort(neutrino_masses)


            L,M,H=sorted_indices[0],sorted_indices[1],sorted_indices[2]
            transformed_spectra = {}

            transformed_spectra[Flavor.NU_E] =      U[0,H]*shifts[H][Flavor.NU_E] + \
                                                    U[0,L]*shifts[L][Flavor.NU_X] + \
                                                    U[0,M]*shifts[M][Flavor.NU_X]

            transformed_spectra[Flavor.NU_E_BAR] =  U[0,L]*shifts[L][Flavor.NU_E_BAR] + \
                                                    U[0,M]*shifts[M][Flavor.NU_X_BAR] + \
                                                    U[0,H]*shifts[H][Flavor.NU_X_BAR]

            transformed_spectra[Flavor.NU_X]  =0.5*(U[1,H]*shifts[H][Flavor.NU_E] + \
                                                    U[2,H]*shifts[H][Flavor.NU_E] + \
                                                    U[1,L]*shifts[L][Flavor.NU_X] + \
                                                    U[2,L]*shifts[L][Flavor.NU_X] + \
                                                    U[1,M]*shifts[M][Flavor.NU_X] + \
                                                    U[2,M]*shifts[M][Flavor.NU_X])


            transformed_spectra[Flavor.NU_X_BAR]=0.5*(U[1,L]*shifts[L][Flavor.NU_E_BAR] + \
                                                    U[2,L]*shifts[L][Flavor.NU_E_BAR] + \
                                                    U[1,M]*shifts[M][Flavor.NU_X_BAR] + \
                                                    U[2,M]*shifts[M][Flavor.NU_X_BAR] + \
                                                    U[1,H]*shifts[H][Flavor.NU_X_BAR] + \
                                                    U[2,H]*shifts[H][Flavor.NU_X_BAR])



            #print(transformed_spectra[Flavor.NU_E])
            return transformed_spectra



    def get_transformed_spectra_project(self, t, E, distance, neutrino_masses, mass_hierachy):
        """Get neutrino spectra after applying oscillation.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial and oscillated spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial and oscillated spectra.

        distance : float
            distance to the supernova assumed to be in kpc

        neutrino_masses: None or List
            Neutrino masses list [m1,m2,m3]. If None, then used the default SNEWPY
        mass_hierachy: str
            mass_hierachy="NO" for Normal ordering or "IO" for INVERTED ordering
        snmodel_dict : dict

        Returns
        -------
        dict
            Dictionary of transformed spectra, keyed by neutrino flavor.
        """
        #print("This is what I'm using")

        if mass_hierachy == "NO":
            theta12 = 33.44<< u.deg
            theta13 = 8.57 << u.deg
            theta23 = 49.20 << u.deg
            deltaCP = 197  << u.deg
            dm21_2 =  7.42e-5  << u.eV**2
            dm31_2 =  2.517e-3 << u.eV**2

        elif mass_hierachy == "IO":
            theta12 = 33.45 << u.deg
            theta13 = 8.60 << u.deg
            theta23 = 49.30 << u.deg
            deltaCP = 282 << u.deg
            dm21_2 = 7.42e-5 << u.eV**2
            dm32_2 = -2.498e-3 << u.eV**2

        else:
            print(" Incorrect mass hierarchy")
            exit()

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
        self.U_e3=s_13*e_mdeltaCP

        # \nu_\mu row
        self.U_u1 = -c_23*s_12-s_23*s_13*c_12*e_deltaCP
        self.U_u2 = c_23*c_12 -s_23*s_13*s_12*e_deltaCP
        self.U_u3=  s_23*c_13

        # \nu_\tau row
        self.U_t1= s_23*s_12-c_23*s_13*c_12*e_deltaCP
        self.U_t2=-s_23*c_12-c_23*s_13*s_12*e_deltaCP
        self.U_t3= c_23*c_13

        distance = distance << u.kpc  # Assume distance is in Kpc
        distance = distance.to('m') # Convert distance to  m

        # Computing time delay dela_ts
        neutrino_masses = neutrino_masses  << u.eV # Assume the masses are given in #!/usr/bin/env python
        neutrino_masses = neutrino_masses.to("MeV")

                # Creating a new energy grid to have the same shape as the time 't'
        # Going to use the same bounds as the default energy # -*- coding: utf-8 -*-

        energy=np.linspace(E[0], E[-1], t.shape[0])

        #print("Yves", t.shape[0])
        # Avoid division by zero, I'm taking the smallest floating point of numpy
        energy[energy==0] = np.finfo(float).eps * E.unit


        delta_t1=0.5*(distance/aconst.c)*(neutrino_masses[0]/energy)**2
        delta_t2=0.5*(distance/aconst.c)*(neutrino_masses[1]/energy)**2
        delta_t3=0.5*(distance/aconst.c)*(neutrino_masses[2]/energy)**2

        #print("Here",delta_t1, delta_t2,delta_t3)

        # Compute initial flux with time delays
        initialspectra_shited_by_delta_t1 = self.get_initial_spectra(t-delta_t1, E)
        initialspectra_shited_by_delta_t2 = self.get_initial_spectra(t-delta_t2, E)
        initialspectra_shited_by_delta_t3 = self.get_initial_spectra(t-delta_t3, E)

        transformed_spectra = {}

        if mass_hierachy == "NO":

            transformed_spectra[Flavor.NU_E] = (abs(self.U_e3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_E] + \
                                               (abs(self.U_e1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_X] + \
                                               (abs(self.U_e2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_X]


            transformed_spectra[Flavor.NU_E_BAR] = (abs(self.U_e1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_E_BAR] + \
                                                   (abs(self.U_e2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_X_BAR] + \
                                                   (abs(self.U_e3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_X_BAR]



            transformed_spectra[Flavor.NU_X] = 0.5*(abs(self.U_u3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_E] + \
                                               0.5*(abs(self.U_t3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_E] + \
                                               0.5*(abs(self.U_u1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_X] + \
                                               0.5*(abs(self.U_t1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_X] + \
                                               0.5*(abs(self.U_u2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_X] + \
                                               0.5*(abs(self.U_t2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_X]

            transformed_spectra[Flavor.NU_X_BAR] = 0.5*(abs(self.U_u1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_E_BAR] + \
                                                   0.5*(abs(self.U_t1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_E_BAR] + \
                                                   0.5*(abs(self.U_u2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_X_BAR] + \
                                                   0.5*(abs(self.U_t2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_X_BAR] + \
                                                   0.5*(abs(self.U_u3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_X_BAR] + \
                                                   0.5*(abs(self.U_t3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_X_BAR]

        if mass_hierachy == "IO":

            transformed_spectra[Flavor.NU_E] = (abs(self.U_e2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_E] + \
                                               (abs(self.U_e1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_X] + \
                                               (abs(self.U_e3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_X]


            transformed_spectra[Flavor.NU_E_BAR] = (abs(self.U_e3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_E_BAR] + \
                                                   (abs(self.U_e1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_X_BAR] + \
                                                   (abs(self.U_e2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_X_BAR]


            transformed_spectra[Flavor.NU_X] = 0.5*(abs(self.U_u2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_E] + \
                                               0.5*(abs(self.U_t2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_E] + \
                                               0.5*(abs(self.U_u1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_X] + \
                                               0.5*(abs(self.U_t1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_X] + \
                                               0.5*(abs(self.U_u3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_X] + \
                                               0.5*(abs(self.U_t3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_X]

            transformed_spectra[Flavor.NU_X_BAR] = 0.5*(abs(self.U_u3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_E_BAR] + \
                                                   0.5*(abs(self.U_t3)**2)*initialspectra_shited_by_delta_t3[Flavor.NU_E_BAR] + \
                                                   0.5*(abs(self.U_u1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_X_BAR] + \
                                                   0.5*(abs(self.U_t1)**2)*initialspectra_shited_by_delta_t1[Flavor.NU_X_BAR] + \
                                                   0.5*(abs(self.U_u2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_X_BAR] + \
                                                   0.5*(abs(self.U_t2)**2)*initialspectra_shited_by_delta_t2[Flavor.NU_X_BAR]


        #print("Deltat1", t-delta_t1,np.finfo(float).eps)

        return transformed_spectra


    def get_flux_project(self, t, E, distance, neutrino_masses, mass_hierachy):
        """Get neutrino flux through 1cm^2 surface at the given distance

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate the neutrino spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the the neutrino spectra.
        distance : astropy.Quantity or float (in kpc)
            Distance from supernova.
        neutrino_masses: None or List
            Neutrino masses list [m1,m2,m3]. If None, then used the default SNEWPY
        mass_hierachy: str
            mass_hierachy="NO" for Normal ordering or "IO" for INVERTED ordering


        Returns
        -------
        dict
            Dictionary of neutrino fluxes in [neutrinos/(cm^2*erg*s)],
            keyed by neutrino flavor.

        """
        distance = distance << u.kpc #assume that provided distance is in kpc, or convert
        factor = 1/(4*np.pi*(distance.to('cm'))**2)
        f = self.get_transformed_spectra_project(t, E, distance, neutrino_masses, mass_hierachy)

        array = np.stack([f[flv] for flv in sorted(Flavor)])
        return  Flux(data=array*factor, flavor=np.sort(Flavor), time=t, energy=E)


    def get_flux_project_arbitrary_masses(self, t, E, distance, neutrino_masses):
        """Get neutrino flux through 1cm^2 surface at the given distance

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate the neutrino spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the the neutrino spectra.
        distance : astropy.Quantity or float (in kpc)
            Distance from supernova.
        neutrino_masses: None or List
            Neutrino masses list [m1,m2,m3]. If None, then used the default SNEWPY

        Returns
        -------
        dict
            Dictionary of neutrino fluxes in [neutrinos/(cm^2*erg*s)],
            keyed by neutrino flavor.

        """
        distance = distance << u.kpc #assume that provided distance is in kpc, or convert
        factor = 1/(4*np.pi*(distance.to('cm'))**2)
        f = self.get_transformed_spectra_project_arbitrary_masses(t, E, distance, neutrino_masses)

        array = np.stack([f[flv] for flv in sorted(Flavor)])
        return  Flux(data=array*factor, flavor=np.sort(Flavor), time=t, energy=E)



    def get_oscillatedspectra(self, *args):
        """DO NOT USE! Only for backward compatibility!

        :meta private:
        """
        warn("Please use `get_transformed_spectra()` instead of `get_oscillatedspectra()`!", FutureWarning)
        return self.get_transformed_spectra(*args)

def get_value(x):
    """If quantity x has is an astropy Quantity with units, return just the
    value.

    Parameters
    ----------
    x : Quantity, float, or ndarray
        Input quantity.

    Returns
    -------
    value : float or ndarray

    :meta private:
    """
    if type(x) == Quantity:
        return x.value
    return x

class PinchedModel(SupernovaModel):
    """Subclass that contains spectra/luminosity pinches for supernova models."""

    def __init__(self, simtab, QCD_effect_time, BH_effect_time, metadata):
        """
        Initialize the PinchedModel using the data from the given table.

        Parameters
        ----------
        simtab: astropy.Table
            Should contain columns TIME, {L,E,ALPHA}_NU_{E,E_BAR,X,X_BAR}.
            The values for X_BAR may be missing, in which case NU_X data will be used.

        QCD_effect_time : float, optional
            Time in seconds when QCD effects are considered in the supernova model. Default is -1 (no QCD effects).
        BH_effect_time : float, optional
            Time in seconds when black hole formation effects are considered in the supernova model. Default is -1 (no black hole effects).

        metadata: dict
            Dictionary containing model parameters.
        """

        # Check and duplicate NU_X data if NU_X_BAR data is missing
        if 'L_NU_X_BAR' not in simtab.colnames:
            for val in ['L', 'E', 'ALPHA']:
                simtab[f'{val}_NU_X_BAR'] = simtab[f'{val}_NU_X']

        # Get grid of model times
        time = simtab['TIME'] << u.s


        # Initialize dictionaries for luminosity, mean energy, and pinch parameter
        self.luminosity = {}
        self.meanE = {}
        self.pinch = {}

        # High scaling values computed from FIG.1 of https://arxiv.org/pdf/2208.14469
        high_scaling_E = {}
        high_scaling_L = {}

        # Determine whether to add QCD effects
        if QCD_effect_time <= 0.:
            high_scaling_L = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0}
            high_scaling_E = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0}
            index=0 # It does not actually matter
            #print("Here")
        else:
            #print("There")
            # Get the index where to insert QCD effects
            index = np.where(time >= QCD_effect_time.to(u.s))[0][0]

            # This Setting are tailored for Bollig 2016 model, find the correct values for the other model if want to use.
            high_scaling_L = {"0": 0.32, "1": 0.9, "2": 1.16, "3": 0.9}
            high_scaling_E = {"0": 0.45, "1": 1.3, "2": 0.6, "3": 1.45}

        # Iterate through each flavor and set the corresponding properties
        for i, f in enumerate(Flavor):
            self.luminosity[f] = simtab[f'L_{f.name}']# << u.erg / u.s
            self.meanE[f] = simtab[f'E_{f.name}']# << u.MeV
            self.pinch[f] = simtab[f'ALPHA_{f.name}']
            if i == 0:
                normalization_L = np.max(self.luminosity[f])
                normalization_E = np.max(self.meanE[f])

            mu = time.value[index]  # Mean
            sigma = 0.0005  # Standard deviation
            height_L = high_scaling_L[str(i)] * normalization_L  # Maximum height
            height_E = high_scaling_E[str(i)] * normalization_E  # Maximum height

            # Calculate the Gaussian distribution
            gaussian_L = height_L * norm.pdf(time.value, mu, sigma) / np.max(norm.pdf(time.value, mu, sigma))
            gaussian_E = height_E * norm.pdf(time.value, mu, sigma) / np.max(norm.pdf(time.value, mu, sigma))

            # Apply Gaussian scaling to luminosity (commented out for specific models)
            self.luminosity[f] += gaussian_L
            self.meanE[f] += gaussian_E

            self.luminosity[f] =self.luminosity[f] << u.erg/u.s
            self.meanE[f] = simtab[f'E_{f.name}'] << u.MeV

        if BH_effect_time > 0.:
            new_meanE = {}
            new_luminosity = {}
            new_pinch = {}
            high_scaling_E = {"0": 28, "1": 23, "2": 32, "3": 23}

            index_BH = np.where(time >= BH_effect_time.to(u.s))[0][0]

            def custom_sigmoid(t, a, b, t1, w):
                w = w * u.ms
                w = w.to(u.s)
                k = 10 / w
                return a + (b - a) / (1 + np.exp(-k * (t - t1 - w / 2)))

            w = 0.5
            t1 = time[index_BH]
            t_values = np.arange(t1.value, time[index_BH+2].value, 0.00001) * time.unit  # Generating new time values
            time = np.concatenate((time[:index_BH], t_values, time[index_BH+2:]))
            self.t_values = t_values

            for i, f in enumerate(Flavor):
                a_E = self.meanE[f][index_BH]  # Starting value
                b_E = high_scaling_E[str(i)] * self.meanE[f].unit  # Ending value

                a_L = self.luminosity[f][index_BH]
                b_L = 0.0*self.luminosity[f].unit

                a_P = self.pinch[f][index_BH]
                b_P = self.pinch[f][index_BH]


                new_meanE[f] = custom_sigmoid(t_values, a_E, b_E, t1, w)
                new_luminosity[f] = custom_sigmoid(t_values, a_L,b_L, t1, w)
                new_pinch[f] = custom_sigmoid(t_values, a_P,b_P, t1, w)  # This remains constants for now.

                self.meanE[f]      = np.concatenate((self.meanE[f][:index_BH], new_meanE[f], b_E*(1+0*self.meanE[f][index_BH+2:].value)))
                self.luminosity[f] = np.concatenate((self.luminosity[f][:index_BH], new_luminosity[f], b_L*self.luminosity[f][index_BH+2:].value))
                self.pinch[f] = np.concatenate((self.pinch[f][:index_BH], new_pinch[f],self.pinch[f][index_BH+2:]))


                #self.luminosity[f][index_BH:] = 0 * self.luminosity[f].unit # Setting the luminosity to 0.0 erg/s

        # Initialize the superclass with the time, QCD_effect_time, and metadata
        super().__init__(time, QCD_effect_time, BH_effect_time, metadata)


    def get_initial_spectra_arbitrary(self, t, E, mass, distance, flavors=Flavor):

        """
        Get initial neutrino spectra/luminosity curves before oscillation.

        Parameters
        ----------
        t : astropy.Quantity
            Time array to evaluate initial spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial spectra.
        mass : astropy.Quantity
            Mass of the neutrino.
        distance : astropy.Quantity
            Distance to the neutrino source.
        flavors : iterable of snewpy.neutrino.Flavor, optional
            List of neutrino flavors to return spectra for (default is all flavors).

        Returns
        -------
        initialspectra : dict
            Dictionary containing the initial spectra for each specified flavor,
            keyed by neutrino flavor.
        """
        # Ensure distance is in meters
        distance = distance.to('m')
        # Ensure mass is in MeV
        mass = mass.to('MeV')

        # Initialize the dictionary to hold the spectra for each flavor
        initialspectra = {}

        # Iterate over the specified neutrino flavors
        for flavor in flavors:
            # Create an array to store spectra for each time-energy pair
            new_result = np.zeros((t.size, E.size))

            # Calculate the time delay due to neutrino mass for each energy
            delta_t = 0.5 * (distance / aconst.c) * (mass / E)**2

            # Iterate over each energy value
            for i, energy in enumerate(E):
                # Adjust time for the mass effect and get the initial spectra
                result = self.get_initial_spectra(t=t - delta_t[i], E=energy)[flavor]
                # Assign the result to the corresponding column
                new_result[:, i] = result

            # Store the spectra in the dictionary with the appropriate units
            initialspectra[flavor] = new_result * result.unit

        return initialspectra


    def get_initial_spectra(self, t, E, flavors=Flavor):
        """Get neutrino spectra/luminosity curves before oscillation.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial spectra.
        flavors: iterable of snewpy.neutrino.Flavor
            Return spectra for these flavors only (default: all)

        Returns
        -------
        initialspectra : dict
            Dictionary of model spectra, keyed by neutrino flavor.
        """
        #convert input arguments to 1D arrays
        t = u.Quantity(t, ndmin=1)
        E = u.Quantity(E, ndmin=1)
        #Reshape the Energy array to shape [1,len(E)]
        E = np.expand_dims(E, axis=0)

        initialspectra = {}

        # Avoid division by zero in energy PDF below.
        E[E==0] = np.finfo(float).eps * E.unit
        #print("KKB",np.finfo(float).eps * E.unit,E[0])

        # Estimate L(t), <E_nu(t)> and alpha(t). Express all energies in erg.
        E = E.to_value('erg')

        # Make sure input time uses the same units as the model time grid, or
        # the interpolation will not work correctly.
        t = t.to(self.time.unit)
        #print("TIME", t)

        for flavor in flavors:
            #print("Bokala",E)
            # Use np.interp rather than scipy.interpolate.interp1d because it
            # can handle dimensional units (astropy.Quantity).
            L  = get_value(np.interp(t, self.time, self.luminosity[flavor].to('erg/s')))
            Ea = get_value(np.interp(t, self.time, self.meanE[flavor].to('erg')))
            a  = np.interp(t, self.time, self.pinch[flavor])

            #Reshape the time-related arrays to shape [len(t),1]
            L  = np.expand_dims(L, axis=1)
            Ea = np.expand_dims(Ea,axis=1)
            a  = np.expand_dims(a, axis=1)
            # For numerical stability, evaluate log PDF and then exponentiate.
            result = \
              np.exp(np.log(L) - (2+a)*np.log(Ea) + (1+a)*np.log(1+a)
                    - loggamma(1+a) + a*np.log(E) - (1+a)*(E/Ea)) / (u.erg * u.s)
            #remove bad values
            result[np.isnan(result)] = 0
            #remove unnecessary dimensions, if E or t was scalar:
            result = np.squeeze(result)
            initialspectra[flavor] = result
            #print("Result",result.shape)
        return initialspectra


class _GarchingArchiveModel(PinchedModel):
    """Subclass that reads models in the format used in the
    `Garching Supernova Archive <https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/>`_."""
    def __init__(self, filename, QCD_effect_time, BH_effect_time, eos='LS220', metadata={}):
        """Model Initialization.

        Parameters
        ----------
        filename : str
            Absolute or relative path to file with model data, we add nue/nuebar/nux.  This argument will be deprecated.
        eos: str
            Equation of state. Valid value is 'LS220'. This argument will be deprecated.

        Other Parameters
        ----------------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        Raises
        ------
        FileNotFoundError
            If a file for the chosen model parameters cannot be found
        ValueError
            If a combination of parameters is invalid when loading from parameters
        """
        if not metadata:
            metadata = {
                'Progenitor mass': float(os.path.basename(filename).split('s')[1].split('c')[0]) * u.Msun,
                'EOS': eos,
            }

        # Read through the several ASCII files for the chosen simulation and
        # merge the data into one giant table.
        mergtab = None
        for flavor in Flavor:
            _flav = Flavor.NU_X if flavor == Flavor.NU_X_BAR else flavor
            _sfx = _flav.name.replace('_', '').lower()
            _filename = '{}_{}_{}'.format(filename, eos, _sfx)
            _lname = 'L_{}'.format(flavor.name)
            _ename = 'E_{}'.format(flavor.name)
            _e2name = 'E2_{}'.format(flavor.name)
            _aname = 'ALPHA_{}'.format(flavor.name)

            # Open the requested filename using the model downloader.
            datafile = _model_downloader.get_model_data(self.__class__.__name__, _filename)

            simtab = Table.read(datafile,
                                names=['TIME', _lname, _ename, _e2name],
                                format='ascii')
            simtab['TIME'].unit = 's'
            simtab[_lname].unit = '1e51 erg/s'
            simtab[_aname] = (2*simtab[_ename]**2 - simtab[_e2name]) / (simtab[_e2name] - simtab[_ename]**2)
            simtab[_ename].unit = 'MeV'
            del simtab[_e2name]

            if mergtab is None:
                mergtab = simtab
            else:
                mergtab = join(mergtab, simtab, keys='TIME', join_type='left')
                mergtab[_lname].fill_value = 0.
                mergtab[_ename].fill_value = 0.
                mergtab[_aname].fill_value = 0.
        simtab = mergtab.filled()
        super().__init__(simtab,QCD_effect_time, BH_effect_time,metadata)


class _RegistryModel(ABC):
    """Base class for supernova model classes that initialise from physics parameters."""

    _param_validator = None

    @classmethod
    def get_param_combinations(cls):
        """Returns all valid combinations of parameters for a given SNEWPY register model.

        Subclasses can provide a Callable `cls._param_validator` that takes a combination of parameters
        as an argument and returns True if a particular combinations of parameters is valid.
        If None is provided, all combinations are considered valid.

        Returns
        -------
        valid_combinations: tuple[dict]
            A tuple of all valid parameter combinations stored as Dictionaries
        """
        for key, val in cls.param.items():
            if not isinstance(val, (list, Quantity)):
                cls.param[key] = [val]
            elif isinstance(val, Quantity) and val.size == 1:
                try:
                    # check if val.value is iterable, e.g. a list or a NumPy array
                    iter(val.value)
                except:
                    cls.param[key] = [val.value] * val.unit
        combos = tuple(dict(zip(cls.param, combo)) for combo in it.product(*cls.param.values()))
        return tuple(c for c in filter(cls._param_validator, combos))

    def check_valid_params(cls, **user_params):
        """Checks that the model-specific values, units, names and conbinations of requested parameters are valid.

        Parameters
        ----------
        user_params : varies
            User-requested model parameters to be tested for validity.
            NOTE: This must be provided as kwargs that match the keys of cls.param

        Raises
        ------
        ValueError
            If invalid model parameters are provided based on units, allowed values, etc.
        UnitTypeError
            If invalid units are provided for a model parameter

        See Also
        --------
        snewpy.models.ccsn
        snewpy.models.presn
        """
        # Check that the appropriate number of params are provided
        if not all(key in user_params for key in cls.param.keys()):
            raise ValueError(f"Missing parameter! Expected {cls.param.keys()} but was given {user_params.keys()}")

        # Check parameter units and values
        for (key, allowed_params), user_param in zip(cls.param.items(), user_params.values()):

            # If both have units, check that the user param value is valid. If valid, continue. Else, error
            if type(user_param) == Quantity and type(allowed_params) == Quantity:
                if get_physical_type(user_param.unit) != get_physical_type(allowed_params.unit):
                    raise UnitTypeError(f"Incorrect units {user_param.unit} provided for parameter {key}, "
                                        f"expected {allowed_params.unit}")

                elif np.isin(user_param.to(allowed_params.unit).value, allowed_params.value):
                    continue
                else:
                    raise ValueError(f"Invalid value '{user_param}' provided for parameter {key}, "
                                     f"allowed value(s): {allowed_params}")

            # If one only one has units, then error
            elif (type(user_param) == Quantity) ^ (type(allowed_params) == Quantity):
                # User param has units, model param is unitless
                if type(user_param) == Quantity:
                    raise ValueError(f"Invalid units {user_param.unit} for parameter {key} provided, expected None")
                else:
                    raise ValueError(f"Missing units for parameter {key}, expected {allowed_params.unit}")

            # Check that unitless user param value is valid. If valid, continue. Else, Error
            elif user_param in allowed_params:
                continue
            else:
                raise ValueError(f"Invalid value '{user_param}' provided for parameter {key}, "
                                 f"allowed value(s): {allowed_params}")

        # Check Combinations (Logic lives inside model subclasses under model.isvalid_param_combo)
        if user_params not in cls.get_param_combinations():
            raise ValueError(
                f"Invalid parameter combination. See {cls.__class__.__name__}.get_param_combinations() for a "
                "list of allowed parameter combinations.")
