# -*- coding: utf-8 -*-
"""
The submodule ``snewpy.models.ccsn`` contains models of core-collapse supernovae
derived from the :class:`SupernovaModel` base class.

You can :ref:`download neutrino fluxes for each of these models <sec-download_models>`
using ``snewpy.get_models("<model_name>")``.

.. _Garching Supernova Archive: https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/
"""
import itertools
import logging
import os
import tarfile

import numpy as np
from astropy import units as u
from astropy.io import ascii
from astropy.table import Table

from snewpy import model_path
from snewpy.models import loaders
from .base import PinchedModel
from .util import check_valid_params, get_param_combinations

class _RegistryModel():
    """TODO: empty base class for now?"""
    pass

class Analytic3Species(PinchedModel):
    """An analytical model calculating spectra given total luminosity,
    average energy, and rms or pinch, for each species.
    """

    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
            Absolute or relative path to file with model data.
        """

        simtab = Table.read(filename,format='ascii')
        self.filename = filename
        super().__init__(simtab, metadata={})


class Nakazato_2013(_RegistryModel):
    """Model based on simulations from Nakazato et al., ApJ S 205:2
    (2013), ApJ 804:75 (2015), PASJ 73:639 (2021). See also http://asphwww.ph.noda.tus.ac.jp/snn/.
    """

    param = {'progenitor_mass': [13, 20, 30, 50] * u.Msun,
             'revival_time': [0, 100, 200, 300] * u.ms,
             'metallicity': [0.02, 0.004],
             'eos': ['LS220', 'shen', 'togashi']}

    _isvalid_combo = lambda p: (p['revival_time'] == 0 * u.ms and p['progenitor_mass'] == 30 * u.Msun and
                                p['metallicity'] == 0.004) or \
                               (p['revival_time'] != 0 * u.ms and p['eos'] == 'shen' and
                                not (p['progenitor_mass'] == 30 * u.Msun and p['metallicity'] == 0.004))
    param_combinations = get_param_combinations(param, _isvalid_combo)

    def __new__(cls, *, progenitor_mass=None, revival_time=None, metallicity=None, eos=None):
        """Model initialization.

        Parameters
        ----------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        revival_time: astropy.units.Quantity
            Time of shock revival in model in units ms. Valid values are {revival_time}.
            Selecting 0 ms will load a black hole formation model
        metallicity: float
            Progenitor metallicity. Valid values are {metallicity}.
        eos: str
            Equation of state. Valid values are {eos}.

        Raises
        ------
        ValueError
            If a combination of parameters is invalid when loading from parameters

        Examples
        --------
        >>> from snewpy.models.ccsn import Nakazato_2013; import astropy.units as u
        >>> Nakazato_2013(progenitor_mass=13*u.Msun, metallicity=0.004, revival_time=0*u.s, eos='togashi')
        Nakazato_2013 Model: nakazato-togashi-BH-z0.004-s30.0.fits
        Progenitor mass  : 30.0 solMass
        EOS              : Togashi
        Metallicity      : 0.004
        Revival time     : 0.0 ms
        """
        # user_params = locals().pop('cls')
        # TODO: Check GitHub PR for error in this example
        # Attempt to load model from parameters

        # Build user params, check validity, construct filename, then load from filename
        user_params = dict(zip(cls.param.keys(), (progenitor_mass, revival_time, metallicity, eos)))
        check_valid_params(cls, **user_params)

        # Store model metadata.
        metadata = {
            'Progenitor mass': progenitor_mass,
            'EOS': eos,
            'Metallicity': metallicity,
            'Revival time': revival_time
        }

        # Strip units for filename construction
        progenitor_mass = progenitor_mass.to(u.Msun).value
        revival_time = revival_time.to(u.ms).value

        if revival_time != 0:
            fname = f"nakazato-{eos}-z{metallicity}-t_rev{int(revival_time)}ms-s{progenitor_mass:3.1f}.fits"
        else:
            fname = f"nakazato-{eos}-BH-z{metallicity}-s{progenitor_mass:3.1f}.fits"

        filename = os.path.join(model_path, cls.__name__, fname)

        if not os.path.isfile(filename):
            # download file from GitHub/Zenodo
            raise NotImplementedError()

        return loaders.Nakazato_2013(filename, metadata)

    # Populate Docstring with param values
    __new__.__doc__ = __new__.__doc__.format(**param)


class Sukhbold_2015(_RegistryModel):
    """Model based on simulations from Sukhbold et al., ApJ 821:38,2016. Models were shared privately by email.
    """
    param = {'progenitor_mass': [27., 9.6] * u.Msun,
             'eos': ['LS220', 'SFHo']}

    param_combinations = get_param_combinations(param)

    def __new__(cls, *, progenitor_mass=None, eos=None):
        """Model Initialization

        Parameters
        ----------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        eos: str
            Equation of state. Valid values are {eos}.

        Raises
        ------
        FileNotFoundError
            If a file for the chosen model parameters cannot be found
        ValueError
            If a combination of parameters is invalid when loading from parameters
        """
        user_params = dict(zip(cls.param.keys(), (progenitor_mass, eos)))
        check_valid_params(cls, **user_params)

        if progenitor_mass.value == 9.6:
            fname = f'sukhbold-{eos}-z{progenitor_mass.value:3.1f}.fits'
        else:
            fname = f'sukhbold-{eos}-s{progenitor_mass.value:3.1f}.fits'

        filename = os.path.join(model_path, cls.__name__, fname)

        # Store model metadata.
        cls.progenitor_mass = float(filename.split('-')[-1].strip('z%.fits')) * u.Msun
        cls.EOS = filename.split('-')[-2]

        if not os.path.isfile(filename):
            # download file from GitHub/Zenodo
            raise NotImplementedError()

        metadata = {
            'Progenitor mass': progenitor_mass,
            'EOS': eos
        }

        return loaders.Sukhbold_2015(filename, metadata)

    # Populate Docstring with param values
    __new__.__doc__ = __new__.__doc__.format(**param)


class Tamborra_2014(_RegistryModel):
    """Model based on 3D simulations from `Tamborra et al., PRD 90:045032, 2014 <https://arxiv.org/abs/1406.0006>`_.
    Data files are from the `Garching Supernova Archive`_.
    """

    param = {'progenitor_mass': [20., 27.] * u.Msun,
             'eos': 'LS220'}
    param_combinations = get_param_combinations(param)

    def __new__(cls,  *, progenitor_mass=None, eos=None):
        check_valid_params(cls, progenitor_mass=progenitor_mass, eos=eos)
        filename = os.path.join(model_path, cls.__name__,
                                f's{progenitor_mass.value:3.1f}c_3D_dir1')

        metadata = {
            'Progenitor mass': progenitor_mass,
            'EOS': eos
        }

        # Metadata is handled by __init__ in _GarchingArchiveModel
        return loaders._GarchingArchiveModel(filename=filename, metadata=metadata)

    # Populate Docstring with param values
    __new__.__doc__ = loaders._GarchingArchiveModel.__init__.__doc__.format(**param)


class Bollig_2016(_RegistryModel):
    """Model based on simulations from `Bollig et al. (2016) <https://arxiv.org/abs/1508.00785>`_. Models were taken, with permission, from the Garching Supernova Archive.
    """

    param = {'progenitor_mass': [11.2, 27.] * u.Msun,
             'eos': 'LS220'}
    param_combinations = get_param_combinations(param)

    def __new__(cls, *, progenitor_mass=None, eos=None):
        check_valid_params(cls, progenitor_mass=progenitor_mass, eos=eos)
        filename = os.path.join(model_path, cls.__name__, f's{progenitor_mass.value:3.1f}c')

        metadata = {
            'Progenitor mass': progenitor_mass,
            'EOS': eos
        }

        return loaders._GarchingArchiveModel(filename=filename, metadata=metadata)

    # Populate Docstring with param values
    __new__.__doc__ = loaders._GarchingArchiveModel.__init__.__doc__.format(**param)


class Walk_2018(_RegistryModel):
    """Model based on SASI-dominated simulations from `Walk et al.,
    PRD 98:123001, 2018 <https://arxiv.org/abs/1807.02366>`_. Data files are from
    the `Garching Supernova Archive`_.
    """

    param = {'progenitor_mass': 15. * u.Msun,
             'eos': 'LS220'}
    param_combinations = get_param_combinations(param)

    def __new__(cls, *, progenitor_mass=None, eos=None):
        check_valid_params(cls, progenitor_mass=progenitor_mass, eos=eos)
        filename = os.path.join(model_path, cls.__name__, f's{progenitor_mass.value:3.1f}c_3D_nonrot_dir1')

        metadata = {
            'Progenitor mass': progenitor_mass,
            'EOS': eos
        }

        return loaders._GarchingArchiveModel(filename=filename, metadata=metadata)

    # Populate Docstring with param values
    __new__.__doc__ = loaders._GarchingArchiveModel.__init__.__doc__.format(**param)


class Walk_2019(_RegistryModel):
    """Model based on SASI-dominated simulations from `Walk et al.,
    PRD 101:123013, 2019 <https://arxiv.org/abs/1910.12971>`_. Data files are
    from the `Garching Supernova Archive`_.
    """

    param = {'progenitor_mass': 40 * u.Msun,
             'eos': 'LS220'}
    param_combinations = get_param_combinations(param)

    def __new__(cls, *, progenitor_mass=None, eos=None):
        check_valid_params(cls, progenitor_mass=progenitor_mass, eos=eos)
        filename = os.path.join(model_path, cls.__name__, f's{progenitor_mass.value:3.1f}c_3DBH_dir1')

        metadata = {
            'Progenitor mass': progenitor_mass,
            'EOS': eos
        }

        return loaders._GarchingArchiveModel(filename=filename, metadata=metadata)

    # Populate Docstring with param values
    __new__.__doc__ = loaders._GarchingArchiveModel.__init__.__doc__.format(**param)


class OConnor_2013(PinchedModel): # TODO: Requires changes to the model file to have one file per model instead of a single gzip archive!
    """Model based on the black hole formation simulation in `O'Connor & Ott (2013) <https://arxiv.org/abs/1207.1100>`_.
    """

    param = {'progenitor_mass': (list(range(12, 34)) +
                                 list(range(35, 61, 5)) +
                                 [70, 80, 100, 120]) * u.Msun,
             'eos': ['HShen', 'LS220']}
    param_combinations = get_param_combinations(param)

    _param_abbrv = {'progenitor_mass': '[12..33, 35..5..60, 70, 80, 100, 120] solMass',
                    'eos': ['HShen', 'LS220']}

    # TODO: This in its changed state will likely break user code -- check this before PR!
    # def __init__(self, base, mass=15, eos='LS220'):  # Previous signature
    def __init__(self, *, progenitor_mass=None, eos=None):
        """Model Initialization.

        Parameters
        ----------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        eos: str
            Equation of state. Valid values are {eos}.

        Raises
        ------
        FileNotFoundError
            If a file for the chosen model parameters cannot be found
        ValueError
            If a combination of parameters is invalid when loading from parameters

        """
        user_params = dict(zip(self.param.keys(), (progenitor_mass, eos)))
        check_valid_params(self, **user_params)
        filename = os.path.join(model_path, self.__class__.__name__, f'{eos}_timeseries.tar.gz')

        # Open luminosity file.
        tf = tarfile.open(filename)

        # Extract luminosity data.
        dataname = 's{:d}_{}_timeseries.dat'.format(int(progenitor_mass.value), eos)
        datafile = tf.extractfile(dataname)
        simtab = ascii.read(datafile, names=['TIME', 'L_NU_E', 'L_NU_E_BAR', 'L_NU_X',
                                             'E_NU_E', 'E_NU_E_BAR', 'E_NU_X',
                                             'RMS_NU_E', 'RMS_NU_E_BAR', 'RMS_NU_X'])

        simtab['ALPHA_NU_E'] = (2.0*simtab['E_NU_E']**2 - simtab['RMS_NU_E']**2)/(simtab['RMS_NU_E']**2 - simtab['E_NU_E']**2)
        simtab['ALPHA_NU_E_BAR'] = (2.0*simtab['E_NU_E_BAR']**2 - simtab['RMS_NU_E_BAR']**2)/(simtab['RMS_NU_E_BAR']**2 - simtab['E_NU_E_BAR']**2)
        simtab['ALPHA_NU_X'] = (2.0*simtab['E_NU_X']**2 - simtab['RMS_NU_X']**2)/(simtab['RMS_NU_X']**2 - simtab['E_NU_X']**2)

        #note, here L_NU_X is already divided by 4
        self.filename = datafile
        self.EOS = eos
        self.progenitor_mass = progenitor_mass

        metadata = {
            'Progenitor mass': self.progenitor_mass,
            'EOS': self.EOS,
        }
        super().__init__(simtab, metadata)

    # Populate Docstring with param values
    __init__.__doc__ = __init__.__doc__.format(**_param_abbrv)


class OConnor_2015(_RegistryModel):
    """Model based on the black hole formation simulation in `O'Connor (2015) <https://arxiv.org/abs/1411.7058>`_.
    """

    param = {'progenitor_mass': 40 * u.Msun,
             'eos': 'LS220'}
    param_combinations = get_param_combinations(param)

    def __new__(cls, *, progenitor_mass=None, eos=None):
        """Model Initialization.

        Parameters
        ----------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        eos: str
            Equation of state. Valid values are {eos}.

        Raises
        ------
        FileNotFoundError
            If a file for the chosen model parameters cannot be found
        ValueError
            If a combination of parameters is invalid when loading from parameters
        """
        user_params = dict(zip(cls.param.keys(), (progenitor_mass, eos)))
        check_valid_params(cls, **user_params)
        # Filename is currently the same regardless of parameters
        filename = os.path.join(model_path, cls.__name__, 'M1_neutrinos.dat')

        metadata = {
            'Progenitor mass': progenitor_mass,
            'EOS': eos,
        }

        return loaders.OConnor_2015(filename, metadata)

    # Populate Docstring with param values
    __new__.__doc__ = __new__.__doc__.format(**param)


class Zha_2021(_RegistryModel):
    """Model based on the hadron-quark phse transition models from `Zha et al. 2021 <https://arxiv.org/abs/2103.02268>`_.
    """

    param = {'progenitor_mass': (list(range(16, 27)) + [19.89, 22.39, 30, 33]) * u.Msun,
             'eos': 'STOS_B145'}
    param_combinations = get_param_combinations(param)

    _param_abbrv = {'progenitor_mass': '[16..26, 19.89, 22.39, 30, 33] solMass',
                    'eos': 'STOS_B145'}

    def __new__(cls, *, progenitor_mass=None, eos=None):
        """Model Initialization.

        Parameters
        ----------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        eos: str
            Equation of state. Valid values are {eos}.

        Raises
        ------
        FileNotFoundError
            If a file for the chosen model parameters cannot be found
        ValueError
            If a combination of parameters is invalid when loading from parameters
        """
        user_params = dict(zip(cls.param.keys(), (progenitor_mass, eos)))
        check_valid_params(cls, **user_params)

        metadata = {
            'Progenitor mass': progenitor_mass,
            'EOS': eos,
        }

        filename = os.path.join(model_path, cls.__name__, f's{progenitor_mass.value:g}.dat')

        return loaders.Zha_2021(filename, metadata)

    # Populate Docstring with abbreviated param values
    __new__.__doc__ = __new__.__doc__.format(**_param_abbrv)


class Warren_2020(_RegistryModel):
    """Model based on simulations from Warren et al., ApJ 898:139, 2020.
    Neutrino fluxes available at https://doi.org/10.5281/zenodo.3667908."""

    # np.arange with decimal increments can produce floating point errors
    # Though it it more intutive to use np.arange, these fp-errors quickly become hindersome
    param = {'progenitor_mass': np.concatenate((np.linspace(9.25, 12.75, 15),
                                                np.linspace(13, 30., 171),
                                                np.linspace(31., 33., 3),
                                                np.linspace(35, 55, 5),
                                                np.linspace(60, 80, 3),
                                                np.linspace(100, 120, 2))) * u.Msun,
             'turbmixing_param': [1.23, 1.25, 1.27],
             'eos': 'SFHo'}
    param_combinations = get_param_combinations(param)

    _param_abbrv = {'progenitor_mass': '[9.25..0.25..13, 13..0.1..30, 31..35, 35..5..60, 70..10..90, 100, 120] solMass',
                    'turbmixing_param': [1.23, 1.25, 1.27],
                    'eos': 'SFHo'}
    # Should turbmixing_param be named 'alpha_lambda'?

    def __new__(cls, progenitor_mass=None, turbmixing_param=None, eos='SFHo'):
        """
        Parameters
        ----------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        turbmixing_param: float
            Turbulent mixing parameter alpha_lambda. Valid Values are {turbmixing_param}
        eos: str
            Equation of state. Valid values are {eos}.

        Raises
        ------
        FileNotFoundError
            If a file for the chosen model parameters cannot be found
        ValueError
            If a combination of parameters is invalid when loading from parameters
        """

        user_params = dict(zip(cls.param.keys(), (progenitor_mass, turbmixing_param, eos)))
        check_valid_params(cls, **user_params)

        fname = f'stir_a{turbmixing_param:3.2f}/stir_multimessenger_a{turbmixing_param:3.2f}_'
        if progenitor_mass.value.is_integer():
            if progenitor_mass.value in (31, 32, 33, 35, 40, 45, 50, 55, 60, 70, 80, 100, 120):
                fname += f'm{int(progenitor_mass.value):d}.h5'
            else:
                fname += f'm{progenitor_mass.value:.1f}.h5'
        else:
            fname += f'm{progenitor_mass.value:g}.h5'
        filename = os.path.join(model_path, cls.__name__, fname)

        # Set model metadata.
        metadata = {
            'Progenitor mass': progenitor_mass,
            'Turb. mixing param.': turbmixing_param,
            'EOS': eos,
        }

        return loaders.Warren_2020(filename, metadata)

    # Populate Docstring with abbreviated param values
    __new__.__doc__ = __new__.__doc__.format(**_param_abbrv)


class Kuroda_2020(_RegistryModel):
    """Model based on simulations from `Kuroda et al. (2020) <https://arxiv.org/abs/2009.07733>`_."""

    param = {'progenitor_mass': 20 * u.Msun,
             'eos': 'LS220',
             'rotational_velocity': [0, 1] * u.rad / u.s,
             'magnetic_field_exponent': [0, 12, 13]}
    _isvalid_combo = lambda p: (p['rotational_velocity'].value == 1 and p['magnetic_field_exponent'] in (12, 13)) or \
                               (p['rotational_velocity'].value == 0 and p['magnetic_field_exponent'] == 0)
    param_combinations = get_param_combinations(param, _isvalid_combo)

    def __new__(cls, progenitor_mass=None, eos=None, rotational_velocity=None, magnetic_field_exponent=None):
        """
        Parameters
        ----------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        eos: str
            Equation of state. Valid values are {eos}.
        rotational_velocity: astropy.units.Quantity
            Rotational velocity of progenitor. Valid values are {rotational_velocity}
        magnetic_field_exponent: int
            Exponent of magnetic field (See Eq. 46). Valid Values are {magnetic_field_exponent}

        Raises
        ------
        FileNotFoundError
            If a file for the chosen model parameters cannot be found
        ValueError
            If a combination of parameters is invalid when loading from parameters
        """
        check_valid_params(cls, progenitor_mass=progenitor_mass, eos=eos, rotational_velocity=rotational_velocity,
                           magnetic_field_exponent=magnetic_field_exponent)
        filename = os.path.join(model_path, cls.__name__,
                                f'LnuR{int(rotational_velocity.value):1d}0B{int(magnetic_field_exponent):02d}.dat')

        metadata = {
            'Progenitor mass': progenitor_mass,
            'EOS': eos,
            'Rotational Velocity': rotational_velocity,
            'B_0 Exponent': magnetic_field_exponent
            }

        return loaders.Kuroda_2020(filename, metadata)

    __new__.__doc__ = __new__.__doc__.format(**param)


class Fornax_2019(_RegistryModel):
    """Model based on 3D simulations from D. Vartanyan, A. Burrows, D. Radice, M.  A. Skinner and J. Dolence, MNRAS 482(1):351, 2019. 
       Data available at https://www.astro.princeton.edu/~burrows/nu-emissions.3d/
    """
    param = {'progenitor_mass': [9, 10, 12, 13, 14, 15, 16, 19, 25, 60] * u.Msun}
    param_combinations = get_param_combinations(param)

    def __new__(cls, progenitor_mass=None, cache_flux=False):
        """Model Initialization.

        Parameters
        ----------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        cache_flux : bool
            If true, pre-compute the flux on a fixed angular grid and store the values in a FITS file.
        """
        metadata = {
            'Progenitor mass': progenitor_mass,
            }

        check_valid_params(cls, progenitor_mass=progenitor_mass)
        if progenitor_mass.value == 16:
            fname = f'lum_spec_{int(progenitor_mass.value):d}M_r250.h5'
        else:
            fname = f'lum_spec_{int(progenitor_mass.value):d}M.h5'
        filename = os.path.join(model_path, cls.__name__, fname)

        return loaders.Fornax_2019(filename, metadata, cache_flux=cache_flux)

    # Populate Docstring with abbreviated param values
    __new__.__doc__ = __new__.__doc__.format(**param)


class Fornax_2021(_RegistryModel):
    """Model based on 3D simulations from D. Vartanyan, A. Burrows, D. Radice, M.  A. Skinner and J. Dolence, MNRAS 482(1):351, 2019. 
       Data available at https://www.astro.princeton.edu/~burrows/nu-emissions.3d/
        """
    param = {'progenitor_mass': (list(range(12, 24)) + [25, 26, 26.99]) * u.Msun}
    param_combinations = get_param_combinations(param)

    _param_abbrv =  {'progenitor_mass': '[12..26, 26.99] solMass'}

    def __new__(cls, progenitor_mass=None):
        """Model Initialization.

        Parameters
        ----------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        """
        check_valid_params(cls, progenitor_mass=progenitor_mass)
        if progenitor_mass.value.is_integer():
            fname = f'lum_spec_{int(progenitor_mass.value):2d}M_r10000_dat.h5'
        else:
            fname = f'lum_spec_{progenitor_mass.value:.2f}M_r10000_dat.h5'
        filename = os.path.join(model_path, cls.__name__, fname)

        metadata = {
            'Progenitor mass': progenitor_mass,
            }

        return loaders.Fornax_2021(filename, metadata)

    # Populate Docstring with abbreviated param values
    __new__.__doc__ = __new__.__doc__.format(**_param_abbrv)

class SNOwGLoBES:
    """A model that does not inherit from SupernovaModel (yet) and imports a group of SNOwGLoBES files."""

    def __init__(self, tarfilename):
        """
        Parameters
        ----------
        tarfilename: str
            Absolute or relative path to tar archive with SNOwGLoBES files.
        """
        self.tfname = tarfilename
        tf = tarfile.open(self.tfname)

        # For now just pull out the "NoOsc" files.
        datafiles = sorted([f.name for f in tf if '.dat' in f.name])
        noosc = [df for df in datafiles if 'NoOsc' in df]
        noosc.sort(key=len)

        # Loop through the noosc files and pull out the number fluxes.
        self.time = []
        self.energy = None
        self.flux = {}
        self.fmin = 1e99
        self.fmax = -1e99

        for nooscfile in noosc:
            with tf.extractfile(nooscfile) as f:
                logging.debug('Reading {}'.format(nooscfile))
                meta = f.readline()
                metatext = meta.decode('utf-8')
                t = float(metatext.split('TBinMid=')[-1].split('sec')[0])
                dt = float(metatext.split('tBinWidth=')[-1].split('s')[0])
                dE = float(metatext.split('eBinWidth=')[-1].split('MeV')[0])

                data = Table.read(f, format='ascii.commented_header', header_start=-1)
                data.meta['t'] = t
                data.meta['dt'] = dt
                data.meta['dE'] = dE

                self.time.append(t)
                if self.energy is None:
                    self.energy = (data['E(GeV)'].data*1000).tolist()

            for flavor in ['NuE', 'NuMu', 'NuTau', 'aNuE', 'aNuMu', 'aNuTau']:
                if flavor in self.flux:
                    self.flux[flavor].append(data[flavor].data.tolist())
                else:
                    self.flux[flavor] = [data[flavor].data.tolist()]

        # We now have a table with rows=times and columns=energies. Transpose
        # so that rows=energy and cols=time.
        for k, v in self.flux.items():
            self.flux[k] = np.transpose(self.flux[k])
            self.fmin = np.minimum(self.fmin, np.min(self.flux[k]))
            self.fmax = np.maximum(self.fmax, np.max(self.flux[k]))

    def get_fluence(self, t):
        """Return the fluence at a given time t.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        fluence : dict
            A dictionary giving fluence at time t, keyed by flavor.
        """
        # Get index of closest element in the array
        idx = np.abs(np.asarray(self.time) - t).argmin()

        fluence = {}
        for k, fl in self.flux.items():
            fluence[k] = fl[:,idx]

        return fluence
