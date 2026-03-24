import numpy as np
import matplotlib.pylab as plt
import yaml
from astropy.io import fits
import sys

from barycorrpy import get_BC_vel
from astropy.time import Time
from astropy.coordinates import SkyCoord


# from pathlib import Path
# file_path = '/Users/ashleybaker/Documents/DLC/object_runs/Altair/'
# config_path = Path(file_path, "obs_config.yaml")
# with open(config_path, 'r') as f:
#     config = yaml.safe_load(f)      
# sys.path.insert(0, str(Path(file_path).parent.parent / "utils"))
# from log_utils import cObsLog   

# # load obs log for reference
# excel_file = str(Path(file_path, "obs_log_original.xlsx"))
# obslog = cObsLog(excel_file)


def get_BC(source_name="MuGem", obstime=None, format='mjd', obsname='Palomar'):
    """
    source_name: star common name
    obstime: observation time string (from TIMEWMJD)
    format: format of obstime for loading with Astropy
    obsname: observatory name

    output
    ------
    bc_vel 
    """
    # Get target coordinates
    target = SkyCoord.from_name(source_name)

    # Your observation
    #obstime = Time('2025-10-20T10:30:00', scale='utc')
    obstime = Time(obstime,format=format)
    # Calculate correction
    bc_vel, warning, status = get_BC_vel(
        JDUTC=obstime.jd,
        ra=target.ra.deg,
        dec=target.dec.deg,
        obsname=obsname,
        ephemeris='de430',
        leap_update=True
    )

    #print(f"Observing {source_name}")
    #print(f"UTC Time: {obstime.iso}")
    #print(f"Barycentric correction: {bc_vel} m/s ({bc_vel/1000} km/s)")

    return bc_vel

def load_all_spectra(target, norder,extension=1,plot=False):
    """
    plot all spectra for a given target and order
    inputs:
    -------
    target - int
        index of target in obslog
    norder - int
        order number to plot
    extension - int
        fits extension to pull data from (1-indexed)
        5 is fiber flat, 1 is extracted flux, 2 is error in extracted flux, 3 is blaze corrected flux
    plot - bool
        whether to plot or not
        
    outputs:
    --------
    wavelengths - 2D array
        wavelengths for each spectrum
    all_flux - 2D array
        flux for each spectrum
    mask_flux - 1D array    
        mask of spectra with high snr only (1 is high snr, 0 is low snr)
    if plot: plot of all spectra for that target and order   
    """
    plt.figure()
    obslog.select_by_target(target)

    # pull out data from f

    Nfiles = len(obslog.obs_data_target['specfile'])
    wavelengths = np.zeros((Nfiles, 2048))
    mask_flux = np.zeros(Nfiles)
    all_flux  = np.zeros((Nfiles, 2048))
    
    for i in np.arange(Nfiles):
        try: # fails bc i didn't download all the data
            # open single fits data
            f = fits.open(config['datapath'] + obslog.obs_data_target['specfile'].values[i])

            # extract wavelength and flux for that order
            wavelengths[i,:] = f[2].data[0,norder,:] 
            flux = f[2].data[extension,norder,:]/np.nanmean(f[2].data[extension,norder,:])

            # plot high snr spectra only    
            avg_flux = np.nanmean( f[2].data[1,norder,:])
            if avg_flux > 5000:
                if plot:
                    plt.plot(wavelengths[i,:], flux, label=f"File {i}")
                all_flux[i,:] = flux
                mask_flux[i] = 1
            else:
                pass
        except:
            pass

    if plot:
        plt.title(f"All spectra for order {norder} - target {target}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Fiber Flat Flux")
        plt.ylim(0,2)
        plt.axvline(x=1282, color='r', linestyle='--', label='Paschen Beta Line (1282 nm)')
        #plt.legend()
        plt.show()

    return  wavelengths, all_flux, mask_flux


