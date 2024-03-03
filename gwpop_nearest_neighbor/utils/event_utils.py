import numpy as np
import pandas as pd
from gwspinpriors import chi_effective_prior_from_isotropic_spins as x_isotropic
import os
import glob
import astropy
Planck15 = astropy.cosmology.Planck15
z_at_value = astropy.cosmology.z_at_value
import h5py
trapz = np.trapz

_BBH_SET = [
    "GW150914", "GW151012", "GW151226", "GW170104", 
    "GW170608", "GW170729", "GW170809", "GW170814", 
    "GW170818", "GW170823", "GW190408_181802", "GW190412", 
    "GW190413_052954", "GW190413_134308", "GW190421_213856", "GW190503_185404", 
    "GW190512_180714", "GW190513_205428", "GW190517_055101", "GW190519_153544", 
    "GW190521", "GW190521_074359", "GW190527_092055", "GW190602_175927", 
    "GW190620_030421", "GW190630_185205", "GW190701_203306", "GW190706_222641", 
    "GW190707_093326", "GW190708_232457", "GW190719_215514", "GW190720_000836", 
    "GW190725_174728", "GW190727_060333", "GW190728_064510", "GW190731_140936", 
    "GW190803_022701", "GW190805_211137", "GW190828_063405", "GW190828_065509", 
    "GW190910_112807", "GW190915_235702", "GW190924_021846", "GW190925_232845", 
    "GW190929_012149", "GW190930_133541", "GW191103_012549", "GW191105_143521", 
    "GW191109_010717", "GW191127_050227", "GW191129_134029", "GW191204_171526", 
    "GW191215_223052", "GW191216_213338", "GW191222_033537", "GW191230_180458", 
    "GW200112_155838", "GW200128_022011", "GW200129_065458", "GW200202_154313", 
    "GW200208_130117", "GW200209_085452", "GW200216_220804", "GW200219_094415", 
    "GW200224_222234", "GW200225_060421", "GW200302_015811", "GW200311_115853", 
    "GW200316_215756"]

def load_posterior_from_meta_file(event_name, catalog_directory, labels=None, verbose=False):
    """
    Modified from gwpopulation_pipe.data_collection
    ========================================================
    Load a posterior from a `PESummary` meta file.

    Parameters
    ----------
    filename: str
    labels: list
        The labels to search for in the file in order of precedence.

    Returns
    -------
    posterior: pd.DataFrame
    meta_data: dict
        Dictionary containing the run label that was loaded.

    """
    _mapping = dict(
        mass_1="mass_1_source",
        mass_2="mass_2_source",
        mass_ratio="mass_ratio",
        redshift="redshift",
        a_1="a_1",
        a_2="a_2",
        cos_tilt_1="cos_tilt_1",
        cos_tilt_2="cos_tilt_2",
    )
    _old_mapping = dict(
        mass_1="m1_detector_frame_Msun",
        mass_2="m2_detector_frame_Msun",
        # mass_ratio="mass_ratio",
        dL="luminosity_distance_Mpc",
        a_1="spin1",
        a_2="spin2",
        cos_tilt_1="costilt1",
        cos_tilt_2="costilt2",
    )

    filename = glob.glob(os.path.join(catalog_directory, rf'*{event_name}*'))[0]
    # print(filename)
    if labels is None:
        labels = ["C01:Mixed"]
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} does not exist")
    _posterior, label = load_meta_file_from_hdf5(filename=filename, labels=labels, verbose=verbose)
    try:
        posterior = pd.DataFrame({key: _posterior[_mapping[key]] for key in _mapping})
    except KeyError:
        posterior = pd.DataFrame({key: _posterior[_old_mapping[key]] for key in _old_mapping})
        posterior['mass_ratio'] = posterior['mass_2'] / posterior['mass_1']
        z_interp = np.linspace(0., 3., 10000)
        dl_interp = Planck15.luminosity_distance(z_interp).to(astropy.units.Mpc).value
        posterior['redshift'] = np.interp(posterior.pop('dL'), dl_interp, z_interp)
        posterior['mass_1'] = posterior['mass_1'] / (1+posterior['redshift'])
        posterior['mass_2'] = posterior['mass_2'] / (1+posterior['redshift'])

    meta_data = dict(label=label)
    print(f"Loaded {label} from {filename}.")
    return posterior, meta_data

def load_meta_file_from_hdf5(filename, labels, verbose=False):
    """
    Modified from gwpopulation_pipe.data_collection
    ========================================================
    Load the posterior from a `hdf5` `PESummary` file.
    See `load_posterior_from_meta_file`.
    """
    new_style = True
    with h5py.File(filename, "r") as data:
        # print(data.keys())
        if "posterior_samples" in data.keys():
            new_style = False
            data = data["posterior_samples"]
        label = list(data.keys())[0]
        for _label in labels:
            if verbose:
                print(data.keys())
            if _label in data.keys():
                label = _label
                break
        if new_style:
            if hasattr(data[label], "keys"):
                if "posterior_samples" in data[label].keys():
                    posterior = pd.DataFrame(data[label]["posterior_samples"][:])
                else:
                    posterior = pd.DataFrame(
                        data[label]["samples"][:],
                        columns=[key.decode() for key in data[label]["parameter_names"][:]],
                    )
            else:
                posterior = pd.DataFrame(data[label][:])
        # print(posterior.columns, label)
        return posterior, label


def load_injection_data(vt_file, ifar_threshold=1, snr_threshold=10, chi_eff=False):
    """
    Slightly modified from gwpopulation_pipe.vt_helper

    Returns format in np.arrays and allows for calculation of chi_eff prior
    =====================================================================

    Load the injection file in the O3 injection file format.

    For mixture files and multiple observing run files we only
    have the full `sampling_pdf`.

    We use a different parameterization than the default so we require a few
    changes.

    - we parameterize the model in terms of primary mass and mass ratio and
      the injections are generated in primary and secondary mass. The Jacobian
      is `primary mass`.
    - we parameterize spins in spherical coordinates, neglecting azimuthal
      parameters. The injections are parameterized in terms of cartesian
      spins. The Jacobian is `1 / (2 pi magnitude ** 2)`.

    For O3 injections we threshold on FAR.
    For O1/O2 injections we threshold on SNR as there is no FAR
    provided by the search pipelines.

    Parameters
    ----------
    vt_file: str
        The path to the hdf5 file containing the injections.
    ifar_threshold: float
        The threshold on inverse false alarm rate in years. Default=1.
    snr_threshold: float
        The SNR threshold when there is no FAR. Default=11.

    Returns
    -------
    gwpop_data: dict
        Data required for evaluating the selection function.

    """
    print(f"Loading VT data from {vt_file}.")

    with h5py.File(vt_file, "r") as ff:
        data = ff["injections"]
        found = np.zeros_like(data["mass1_source"][()], dtype=bool)
        for key in data:
            if ("ifar" in key.lower()) and ('cwb' not in key.lower()):
                found = found | (data[key][()] > ifar_threshold)
            if "name" in data.keys():
                gwtc1 = (data["name"][()] == b"o1") | (data["name"][()] == b"o2")
                found = found | (gwtc1 & (data["optimal_snr_net"][()] > snr_threshold))
        n_found = sum(found)
        print("I found {0} out of {1} injections".format(n_found, len(found)))
        gwpop_data = dict(
            mass_1=np.array(data["mass1_source"][found]),
            mass_ratio=np.array(
                data["mass2_source"][found] / data["mass1_source"][found]
            ),
            redshift=np.array(data["redshift"][found]),
            total_generated=int(data.attrs["total_generated"][()]),
            analysis_time=data.attrs["analysis_time_s"][()] / 365.25 / 24 / 60 / 60,
        )
        for ii in [1, 2]:
            gwpop_data[f"a_{ii}"] = (
                np.array(
                    data.get(f"spin{ii}x", np.zeros(n_found))[found] ** 2
                    + data.get(f"spin{ii}y", np.zeros(n_found))[found] ** 2
                    + data[f"spin{ii}z"][found] ** 2
                )
                ** 0.5
            )
            gwpop_data[f"cos_tilt_{ii}"] = (
                np.array(data[f"spin{ii}z"][found]) / gwpop_data[f"a_{ii}"]
            )
        if chi_eff:
            gwpop_data["chi_eff"] = (
                (gwpop_data["a_1"]*gwpop_data["cos_tilt_1"] 
                + gwpop_data["mass_ratio"]*gwpop_data["a_2"]*gwpop_data["cos_tilt_2"]) 
                / (1+gwpop_data["mass_ratio"])
            )

            qs = gwpop_data["mass_ratio"]
            xs = gwpop_data["chi_eff"]
            chi_eff_prior = x_isotropic(np.array(qs),0.998,np.array(xs)) 
            # https://zenodo.org/record/5546676#.ZEwH4XbMKUk

            chi_eff_prior = np.array(chi_eff_prior)

            gwpop_data["prior"] = np.array(
                np.array(data["sampling_pdf"][found]) # includes p(x1, y1, z1, x2, y2, z2)
                * np.array(data["mass1_source"][found])
                * chi_eff_prior
                * (4 * np.pi * gwpop_data["a_1"] ** 2) 
                * (4 * np.pi * gwpop_data["a_2"] ** 2)   # divide out p(x1, y1, z1, x2, y2, z2)
            )

            gwpop_data["isotropic_chieff_prior"] = chi_eff_prior
            
        else:
            gwpop_data["prior"] = (
                np.array(data["sampling_pdf"][found])
                * np.array(data["mass1_source"][found])
                * (2 * np.pi * gwpop_data["a_1"] ** 2)
                * (2 * np.pi * gwpop_data["a_2"] ** 2)
            )

    return gwpop_data

def euclidean_distance_prior(samples):
    r"""
    Modified from gwpopulation_pipe.data_collection
    ========================================================
    Evaluate the redshift prior assuming a Euclidean universe.

    See Appendix C of `Abbott et al. <https://arxiv.org/pdf/1811.12940.pdf>`_.

    .. math::

        p(z) \propto d^2_L \left( \frac{d_{L}}{1 + z} + (1 + z) \frac{d_{H}}{E(z)} \right)

    This uses the `astropy.cosmology.Planck15` cosmology.

    Parameters
    ----------
    samples: dict
        The samples to use, must contain `redshift` as a key.

    """
    redshift = np.array(samples["redshift"])
    luminosity_distance = Planck15.luminosity_distance(redshift).to(astropy.units.Gpc).value
    p = luminosity_distance**2 * (
        luminosity_distance / (1 + redshift)
        + (1 + redshift)
        * Planck15.hubble_distance.to(astropy.units.Gpc).value
        / Planck15.efunc(redshift)
    )
    return np.array(p)

def reformat(posterior_list, posterior_names, minimum_length=np.inf):
    """
    Modified from gwpopulation_pipe.data_collection
    ========================================================
    Map the keys from legacy names to the `GWPopulation` standards.

    Parameters
    ----------
    posts: dict
        Dictionary of `pd.DataFrame` objects

    Returns
    -------
    new_posts: dict
        Updated posteriors.

    """
    _mapping = dict(
        mass_1="m1_source",
        mass_2="m2_source",
        mass_ratio="q",
        a_1="a1",
        a_2="a2",
        cos_tilt_1="costilt1",
        cos_tilt_2="costilt2",
        redshift="redshift",
    )
    lengths = [len(p['mass_1']) for p in posterior_list]
    args = np.argsort(lengths)
    # print(lengths)
    # print(posterior_names[args], lengths[args])
    _minimum_length = np.min(lengths)
    
    if _minimum_length < minimum_length:
        print(f'Using {_minimum_length} samples from GW posteriors. User asked for {minimum_length} samples')
        minimum_length = _minimum_length
    print('Original numbers of posterior samples are:')
    for ii in range(len(posterior_list)):
        print(f'{posterior_names[ii]}: {lengths[ii]}')
    print(f'Downsampling to {minimum_length} samples from each posterior')

    new = {}
    for ii, post in enumerate(posterior_list):
        post = post.sample(n=minimum_length, ignore_index=True)
        # new = pd.DataFrame()
        for key in _mapping:
            if not key in new:
                new[key] = []
            if _mapping[key] in post:
                new[key].append(post[_mapping[key]])
            elif key in post:
                new[key].append(post[key])
            else:
                print(f'We have an issue with the {key} samples from {posterior_names[ii]}')
                new[key].append([0])
    for key in new:
        new[key] = np.array(new[key])

    new["chi_eff"] = ( (new["a_1"]*new["cos_tilt_1"]
                    + new["mass_ratio"]*new["a_2"]*new["cos_tilt_2"])
                    / (1 + new["mass_ratio"]) )
    return new

def evaluate_prior(posts, posterior_names, max_redshift=1.9, distance_prior='euclidean', 
                   mass_prior="flat-detector", spin_prior="component"):
    """
    Modified from gwpopulation_pipe.data_collection
    ========================================================

    Evaluate the prior distribution for the input posteriors.

    Parameters
    ----------
    posts: dict
        Dictionary of `pd.DataFrame` objects containing the posteriors.
    args:
        Input args containing the prior specification.

    Returns
    -------
    posts: dict
        The input dictionary, modified in place.
    """
    max_redshift = max(
        max_redshift, np.max(posts["redshift"])
    )
    zs_ = np.linspace(0, max_redshift * 1.01, 1000)
    if distance_prior.lower() == "comoving":
        print(
            f"Using uniform in the comoving source frame distance prior for all events."
        )
        p_z = Planck15.differential_comoving_volume(zs_).value * 4 * np.pi / (1 + zs_)
    else:
        print("Using Euclidean distance prior for all events.")
        p_z = euclidean_distance_prior(dict(redshift=zs_))
    p_z /= trapz(p_z, zs_)
    # interpolated_p_z = np.interp1d(zs_, p_z)

    posts["prior"] = np.ones_like(posts["redshift"])
    if "redshift" in posts:
        posts["prior"] *= np.interp(posts["redshift"], zs_, p_z)
    else:
        print(
            "No redshift present, cannot evaluate distance prior weight"
        )
    if mass_prior.lower() == "flat-detector":
        print(f"Assuming flat in detector frame mass prior")
        posts["prior"] *= posts["mass_1"] * (1 + posts["redshift"]) ** 2
    else:
        raise ValueError(f"Mass prior {mass_prior} not recognized.")
    if spin_prior.lower() == "component":
        print(f"Assuming uniform in component spin prior")
        posts["prior"] /= 4
    
    elif spin_prior.lower() == "chieff-isotropic":
        print(f"Assuming uniform in component spin and isotropic, with samples in chi-eff")
        chieff_prior = x_isotropic(np.array(posts["mass_ratio"]), 0.99, np.array(posts["chi_eff"]))
        posts["isotropic_chieff_prior"] = chieff_prior
        posts["prior"] *= chieff_prior
    else:
        raise ValueError(f"Spin prior {spin_prior} not recognized.")
    return posts


def load_gwtc3pop_bbh_set(catalog_directory, labels=['C01:Mixed'], minimum_length=np.inf, chi_eff=False, event_list=_BBH_SET, special_events=[], special_labels=['C01:IMRPhenomXPHM'], verbose=False):
    '''
    catalog_directory, path to where the event.hdf5s are stored
    labels, list in order of which label to use in order of preference
    minimum_length, length to downsample to
    chieff, whether to calculate p(chieff|PE prior) for each event sample list
    event_list, the list of events to use
    special_events, list of events (in the same naming convention as above) for which to use a special label
    special_labels, list of labels in order of preference to use for special events, e.g. 
    '''
    posterior_list = []
    for bbh in event_list:
        if bbh in special_events:
            posterior, meta_data = load_posterior_from_meta_file(bbh, catalog_directory, labels=special_labels, verbose=verbose)
        else:
            posterior, meta_data = load_posterior_from_meta_file(bbh, catalog_directory, labels=labels)
        posterior_list.append(posterior)

    posterior_list = reformat(posterior_list, event_list, minimum_length=minimum_length)
    if chi_eff:
        chieff_prior = "chieff-isotropic"
    else:
        chieff_prior = "component"

    posterior_list = evaluate_prior(posterior_list, event_list, spin_prior=chieff_prior)
    return posterior_list


