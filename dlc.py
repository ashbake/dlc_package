from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from scipy.ndimage import zoom, shift, rotate
from scipy.optimize import curve_fit

from pathlib import Path

from astropy.convolution import Gaussian1DKernel, convolve
from flatstar import draw
import yaml

from astropy import units as u
from astropy import constants as c


from dataclasses import dataclass, field

FILEPATH = Path(__file__).resolve().parent

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = [10,4]
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.cal'] = 'Helvetica Neue LT Pro'  # changes only the mathcal subfamily


# use starry to make 2d model?
class StellarModel():
    def __init__(self):
        pass
    
    def generate_plane_image(self, slope, angle, width, height):
        """Generates a 2D image of a plane with given slope and angle.
        e.g. for star with spin aligned vertically 
            RV=|v| * cos(theta),   cos(theta) = x / R
            RV=|v| * x / R

        angle here is on sky spin axis w.r.t. fiber offset - not used yet and not sure needed, complicates assumptions made later
        """

        # Create a grid of x and y coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Calculate the z values for the plane
        #z = slope * (x * np.cos(np.radians(angle)) + y * np.sin(np.radians(angle)))
        z = slope * x # force star to always have velocity x aligned to vibe with y collapse later, akin to angle=0

        return z
    
    def generate_star_spot(self):
        """add spots to star"""
        pass

    def load_phoenix(self):
        """load phoenix model of star"""
        pass


class AOJitter():
    def __init__(self,):
        pass

    def generate_jitter(self, tt_dyn):
        """generate a jitter pattern based on the time scale of the jitter and the exposure time"""
        pass

    def DAR(self):
        """compute the DAR shift based on the zenith angle and wavelength range of the observation"""
        pass


@dataclass
class dlcSettings():
    """All settings for DLC class"""
    # INSTRUMENT
    # telescope [str] default: Keck - string name of telescope to load coupling maps for
    #    options: Keck, Hale, or TMT
    telescope: str = 'Keck'

    # R  [float] default: 100000 - resolving power of instrument
    R: float = 100_000

    # OBSERVATION
    # offsets [array | list] default: [0,0]*u.mas- offset PSF to fiber positions in array [x,y] in milliarcseconds 
    xoffsets: list = field(default_factory=lambda: [0, 0] * u.mas) 
    yoffsets: list = field(default_factory=lambda: [0, 0] * u.mas)

    # Jitter
    tt_dyn: float = 0*u.mas

    # STAR
    vsini: float = 2 * u.km / u.s
    theta: float = 0 * u.deg
    diameter: float = 13 * u.mas

    # DAR on/off
    DAR: bool = True

class DifferentialLimbCoupling(dlcSettings,
                               StellarModel,
                               AOJitter):
    """        
    inputs:
    ------
    config [str | dic | None] (default: None)
        config file either a path to a yaml file or a dictionary or None

    diagnostics_on [bool] (default: True)
        if True will plot intermediate files as the code does stuff
    """
    def __init__(self, 
                 config=None, 
                 diagnostics_on=False):
        
        dlcSettings.__init__(self)  # make sure all attributes are applied from settings
        
        # override deafult settings with config file if provided
        if config != None: 
            self._loadConfig(config) 
        
        self.diagnostics_on = diagnostics_on 

        # run through presets to load necessary files and settings for the class
        self._loadCouplingMap()

        # create star
        self._create_star_image()

        # create instrument kernel
        self._create_instrument_kernel()

    def _loadCouplingMap(self):
        """ load coupling map for the telescope and zoom in on it

            loads plate scale for the coupling map
        """
        if self.telescope=='Keck':
            coupling_file_name = 'staticModel_Keck_HK_defoc0nmRMS_LO0nmRMS_Fnum2.63_atm0_adc0_nWvls10_PL0.fits'
        #elif telescope=='Hale':
        #    coupling_map = ""
        elif self.telescope == 'TMT' or self.telescope == 'Thirty Meter Telescope':
            coupling_file_name = 'staticModel_TMT_HK_defoc0nmRMS_LO0nmRMS_Fnum2.61_atm0_adc0_nWvls10_PL0.fits'
        elif self.telescope=='Palomar' or self.telescope=='Hale':
            coupling_file_name = 'staticModel_Keck_yJ_defoc0nmRMS_LO0nmRMS_Fnum3.67_atm0_adc0_nWvls10_PL0.fits'
            # use keck one for now, but should make a palomar specific one in the future, just use the keck one and scale it to match the plate scale of palomar since the coupling map is in pixel units
        else:
            raise(Warning, 'Telescope Data not available, options are: Keck, TMT, Palomar, Hale')
        
        coupling_map_path = FILEPATH / "data/coupling_maps/" / coupling_file_name
        print(f'Loading {coupling_map_path}')
        map_ext = 7 # coupling map extension (wavelength)

        # make wavelength array for each layer of coupling map cube
        wstart = fits.getval(coupling_map_path, 'LAMMIN')
        wend = fits.getval(coupling_map_path, 'LAMMAX')
        nwvls = fits.getval(coupling_map_path, 'NAXIS3')
        warr_fits = np.linspace(wstart, wend, nwvls)

        self.plate_scale_coupling_raw = fits.getval(coupling_map_path, 'DXMAS') * u.mas / u.pixel# mas per pixel
        if self.telescope=='Palomar' or self.telescope=='Hale':
            # scale the plate scale to match palomar's plate scale since using keck coupling map
            self.plate_scale_coupling_raw *= 3.5 # scale factor to match palomar's plate scale, should make a palomar specific coupling map in the future

        # image midpoint
        midpix = int(fits.getval(coupling_map_path, 'NAXIS1') / 2.)

        # zoom in image size for coupling map
        wz = 200

        # load in central chunk of coupling map (makes the image processing faster later)
        self.coupling_map_raw = fits.getdata(coupling_map_path)[map_ext][midpix-wz:midpix+wz, midpix-wz:midpix+wz]

        # upsample the coupling map by a factor of N
        upsamp_fac = 10 # upsample factor
        self.coupling_map = zoom(self.coupling_map_raw , upsamp_fac, cval=0,order=3,grid_mode=False)
        self.gridsize = self.coupling_map.shape[0]

        #print(np.argmax(np.nansum(coupling_map_upsampled,axis=1)))
        #print(np.shape(coupling_map_upsampled))

        # new plate scale after upsampling
        self.plate_scale_coupling = self.plate_scale_coupling_raw / upsamp_fac

        if self.diagnostics_on:
            plt.imshow(self.coupling_map, origin='lower')
            plt.grid()
            plt.title(f'Coupling Map, Plate Scale: {np.round(self.plate_scale_coupling,4)}')
            plt.show()

    def _loadConfig(self, config):
        """
        Load config with all settings if provided. Otherwise use defaults in dlcSettings dataclass.
 
        input
        ------
        config [str | dic | None] (default: None)
            config file either a path to a yaml file or a dictionary or None
        """
        # load yaml file or dictionary
        if isinstance(config, str):
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f) # load yaml file if config is a string
        elif isinstance(config, dict):
            config_dict = config # use config directly if it's already a dictionary
        else:
            raise ValueError("Config must be either a path to a yaml file or a dictionary.")
        
        # set attributes in class based on config file, if not provided use defaults from dataclass
        for key, value in config_dict.items():
            if hasattr(self, key):
                # get current attribute from default to get units
                current_attr = getattr(self, key)
                setattr(self, key, value)
                
                # inherit the units if the provided value doesn't have units but the default does
                if hasattr(current_attr, 'unit'):
                    if not hasattr(value, 'unit'):
                        setattr(self, key, value * current_attr.unit)
                    else:
                        assert value.unit==current_attr.unit, f"Units for {key} do not match default units. Expected {current_attr.unit}, got {value.unit}"
    
    @u.quantity_input
    def _compute_shift(self, 
                       xoffset: u.mas,
                       yoffset: u.mas,
                       theta: u.deg):
        """ translate the shift in x,y to angular deg to apply to coupling map
        in a way that utilizes the fiber symmetry
 
        inputs
        ------
        xoffset,yoffset - offset position of star w.r.t. fiber (u.mas)
        theta - angle of stellar axis w.r.t. xy plane (u.deg)

        returns
        -------
        xshift_pix - how much to shift coupling map in x (axis 1)
        yshift_pix - how much to shift coupling map in y (axis 0)
        """
        # define x, y to be offset of coupling map w.r.t fiber, so need to flip sign of offsets
        x = -xoffset 
        y = -yoffset

        shifted_offset   = np.sqrt(x**2 + y**2)  # mas, from center of field
        if x==0: 
            phi=90 * u.deg
        else:
            phi = np.arctan(y/x) # bc x and y have units, returns radians already

        if x < 0:
            phi += 180 * u.deg # arctan only gives angles between -90 and 90, need to add 180 if in left half of field
        
        angle            = phi + theta           # degrees, shifts coupling map

        shifted_offset_pix = shifted_offset  * 1./ self.plate_scale_coupling   # pixels
        xshift_pix = shifted_offset_pix.decompose().value * np.cos(angle)
        yshift_pix = shifted_offset_pix.decompose().value * np.sin(angle)

        return xshift_pix, yshift_pix
    
    def _transform_coupling_map(self, coupling_map, xshift_pix, yshift_pix):
        """ apply the shift to the coupling map based on the shifted offset and angle
        """
        # compute the desired shift in image pixel units
        
        # shift coupling map accordingly
        # shift does FFT but is super slow
        #coupling_map_shifted = shift(coupling_map, [xshift_pix, yshift_pix],mode='wrap')
        
        # try rolling instead
        coupling_map_shifted = np.roll(coupling_map, xshift_pix, axis=1) # really important apply these to the correct axis
        coupling_map_shifted = np.roll(coupling_map_shifted, yshift_pix, axis=0)

        # make it so coupling maps have same peak value (which gets messed up when shifting)
        unshifted_sum_coupling = np.max(self.coupling_map)
        coupling_map_shifted *= (unshifted_sum_coupling / np.max(coupling_map_shifted))

        return coupling_map_shifted
    
    def _create_star_image(self, ):
        """
         create a limb darkened star image with velocity map based on vsini and diameter of star

         returns:
            im_intensity - 2D array of limb darkened star intensity
            vel_arr - 1D array of velocity values corresponding to the velocity map
            stellar_im_vel - 2D array of the velocity map of the star, scaled by the limb darkened intensity and mask
            vel_scale_pix - velocity scale in km/s per pixel for the velocity map
         """
        # compute the stellar diameter in the pixel units for the coupling image 
        radius_star_pix = (self.diameter / 2.) * 1./ self.plate_scale_coupling # pixels
        radius_star_norm = radius_star_pix.decompose().value / self.gridsize

        # stellar vsini array
        vel_scale_pix =  self.vsini / (radius_star_pix * 2) # km/s per pixel in the image
        vel_arr = np.linspace(-vel_scale_pix * self.gridsize, vel_scale_pix * self.gridsize,num=self.gridsize) 

        # velocity to pixel slope in image TODO check this
        vel_slope = self.vsini * 2. / (self.gridsize * radius_star_norm)
        print('vel slope', vel_slope)

        ####### LIMB DARKENING
        # generate a limb darkened stellar image
        im_limb_dark = draw.star(self.gridsize, radius=radius_star_norm, limb_darkening_law='quadratic',ld_coefficient=[0.9,0.4])

        # generate a binary mask stellar image (used to clean up some edge effects)
        im_mask = draw.star(self.gridsize,radius=radius_star_norm * 0.97, limb_darkening_law='quadratic',ld_coefficient=[0.,0.])
        im_mask_int = im_mask.intensity / np.nanmax(im_mask.intensity)

        # mask and normalize the intensity image
        im_intensity = im_limb_dark.intensity / np.nanmax(im_limb_dark.intensity) * im_mask_int

        ######## 2D VELOCITY IMAGE
        # generate 2D velocity image based on the stellar vsini and disc size
        # force angle to be 0, keep star aligned to x,y plane bc will collapse it in y later
        plane_image = self.generate_plane_image(vel_slope, 0, im_limb_dark.intensity.shape[0], im_limb_dark.intensity.shape[1])
        plane_image -= np.nanmean(plane_image)

        # scale the limb-darkened image
        stellar_im_vel = plane_image * im_mask_int * im_intensity
        stellar_im_vel /= np.nanmax(stellar_im_vel)
        
        # store important variables
        self.im_intensity, self.vel_arr, self.stellar_im_vel, self.vel_scale_pix = im_intensity, vel_arr, stellar_im_vel, vel_scale_pix
    
    def _create_instrument_kernel(self,):
        """
        section for generating the spectrograph PSF convolution kernel
        saved in self.spec_kernel
        """
        spec_res_vel =  c.c / self.R # spectrometer resolution 
        spec_kernel_wid = spec_res_vel / self.vel_scale_pix # PSF FWHM in image pixels
        spec_kernel_sigma = spec_kernel_wid.decompose() / 2.335 # sigma of gaussian in pixels
        self.spec_kernel = Gaussian1DKernel(spec_kernel_sigma.value).array # final kernel
        print('res vel', np.round(spec_res_vel,1), 'spec kernel wid', np.round(spec_kernel_wid,1))

    @u.quantity_input
    def _create_lsf(self, coupling_map):
        """
        Shift the coupling map and apply the stellar image to compute the
          reference and shifted line spread profiles with their CCFs

        inputs
        ------
        coupling_map - the coupling map, already shifted if relevant
       
        returns
        -------
        vprof - the integrated line profile for the reference position
        vprof_shifted - the integrated line profile for the shifted position
        ccf_ref - the line profile convolved with a fake spectrometer PSF for the reference position
        ccf_shifted - the line profile convolved with a fake spectrometer PSF for the shifted position
        """
        # Create Integrated Line Profiles
        scale_fac    = np.nanmax(np.nansum(self.im_intensity * coupling_map, axis=0))
        vprof = np.nansum(self.im_intensity * coupling_map, axis=0)
        vprof /= scale_fac
        vprof = (1. - vprof)

        # convolve the integrated line profil with the fake spectrometer PSF kernel
        ccf_shifted  = convolve(vprof,self.spec_kernel,normalize_kernel=True, boundary='extend',fill_value=0)

        # these returns are computer per offset position, so don't store individual ones in class
        return vprof, ccf_shifted

    def _find_ccf_min(self, v, ccf, method='fit'):
        """
        Find the minimum of the CCF to determine the velocity shift for a given position. This is the RV shift that would be measured for this position.

        inputs
        ------
        v - velocity array corresponding to the CCF
        ccf - the cross-correlation function for the line profile convolved with the spectrometer LSF
        method - method to find minimum, either 'fit' to fit a parabola to the minimum or 'min' to just take the minimum value (default: 'fit')
        """
        # fit ccf vs v to find minimum
        if method=='fit':
            # define a parabola function to fit to the CCF around the minimum
            def parabola(x, a, b, c):
                return a * (x - b)**2 + c

            # find the index of the minimum value in the CCF
            min_index = np.argmin(ccf)

            # fit a parabola to the points around the minimum
            fit_indices = np.arange(min_index-2, min_index+3) # take 5 points around the minimum for fitting
            popt, _ = curve_fit(parabola, v[fit_indices], ccf[fit_indices])

            # the vertex of the parabola gives the velocity shift
            a, b, c = popt
            vel_shift = b * v.unit

        # argmin method
        if method=='min':
            min_index = np.argmin(ccf)
            vel_shift = v[min_index]    
        
        return vel_shift

    def run(self):
        """ 
        run the DLC simulation with the current settings in the class
        """
        # get the reference line profile and ccf
        self.vprof_ref, self.ccf_ref = self._create_lsf(self.coupling_map)
        self.vel_ref   = self._find_ccf_min(self.vel_arr, self.ccf_ref) * u.pix
 
        self.ccfs_shifted = {}
        self.vprofs_shifted = {}
        self.coupling_maps_shifted = {}
        vels = np.zeros(len(self.yoffsets)) * u.km / u.s
        for i,yoffset in enumerate(self.yoffsets):
            print('Running x,y position', self.xoffsets[i], yoffset)
            
            # compute shifted coupling map
            shifted_offset, angle = self._compute_shift(self.xoffsets[i],yoffset,self.theta)
            shifted_coupling_map = self._transform_coupling_map(self.coupling_map, shifted_offset, angle)
            
            # compute vprof and ccf for shifted position
            vprof_shifted, ccf_shifted = self._create_lsf(shifted_coupling_map)
            
            # compute min velocity based on ccf
            vel_shift      = self._find_ccf_min(self.vel_arr, ccf_shifted) * u.pix
            print('vel', vel_shift - self.vel_ref)
            
            # store outputs for this position
            self.ccfs_shifted[i] = ccf_shifted
            self.vprofs_shifted[i] = vprof_shifted
            self.coupling_maps_shifted[i] = shifted_coupling_map
            vels[i]   = vel_shift - self.vel_ref

        self.vels = vels # don't't need the pixels unit

        if self.diagnostics_on:
            self.plot_rvs()
            self.plot_star_fiber()
            #self.plot_star_fiber_RValigned()
            #self.plot_star_fiber_raw()
            self.plot_lsf_ccf()

    def plot_star_fiber_raw(self, save_path=None):
        """
        plot the fiber coupling map and overplot the stellar positions with 
        the reference star shown with its velocity map. This is the raw shifted
        coupling maps with the star always in the center.
        """
        fig, axs = plt.subplots(1,3,figsize=[10,4])
        extent = self.plate_scale_coupling * len(self.stellar_im_vel)
        extent = extent.value
    
        axs[0].imshow((self.stellar_im_vel*self.coupling_map  + self.coupling_map),cmap='RdBu_r', origin='lower', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[1].imshow((self.stellar_im_vel*self.coupling_maps_shifted[0]  + self.coupling_maps_shifted[0]),cmap='RdBu_r', origin='lower', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[2].imshow((self.stellar_im_vel*self.coupling_maps_shifted[len(self.xoffsets)-1]  + self.coupling_maps_shifted[len(self.xoffsets)-1]),cmap='RdBu_r', origin='lower', extent=(-extent/2, extent/2, -extent/2, extent/2))

        axs[0].set_xlabel('X [mas]')
        axs[0].set_ylabel('Y [mas]')
        axs[0].set_title('Reference')

        axs[1].set_xlabel('X [mas]')
        axs[1].set_title('First shifted')

        axs[2].set_xlabel('X [mas]')
        axs[2].set_title('Last shifted')

        # plot over reference the position of the xoffsets and yoffsets
        axs[0].plot(self.xoffsets, self.yoffsets, marker='x', color='k', label='Offset positions')
        #axs[0].legend()
        
        plt.suptitle(f'Star Coupling Maps, As Calculated\ntheta={self.theta}')

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_star_fiber_RValigned(self, save_path=None):
        """
        plot the fiber coupling map and overplot the stellar positions with the 
        reference star shown with its velocity map. This shifts the coupling map 
        back to be centered and rotate it so the star is at its correct angle
        """
        fig, axs = plt.subplots(1,3,figsize=[10,4])
        extent = self.plate_scale_coupling * self.coupling_map.shape[0]
        extent = extent.value

        # PLOT PANEL 0 - reference position
        axs[0].imshow((self.stellar_im_vel*self.coupling_map  + self.coupling_map),cmap='RdBu_r', origin='lower', extent=(-extent/2, extent/2, -extent/2, extent/2))
        # rotate xoffset and yoffsets by theta to plot over the reference position
        # the velocity axis is aligned and so the offsets need to be rotated to match the angle of the star
        x_new = self.xoffsets * np.cos(-self.theta) - self.yoffsets * np.sin(-self.theta)
        y_new = self.xoffsets * np.sin(-self.theta) + self.yoffsets * np.cos(-self.theta)
        axs[0].plot(x_new, y_new, marker='x', color='k', label='Offset positions')

        axs[0].set_xlabel('X [mas]')
        axs[0].set_ylabel('Y [mas]')
        axs[0].set_title('Reference')

        def deshift_coupling_map(coupling_map_shifted, xoffset, yoffset, theta):
            """ function to shift the coupling map back to be centered """
            shifted_offset, angle = self._compute_shift(xoffset,yoffset,theta)
            unshifted_map = self._transform_coupling_map(coupling_map_shifted, -1*shifted_offset, angle)
            return unshifted_map
        
        # plot de shifted coupling map
        unshifted_temp_map = np.zeros_like(self.coupling_map)
        for i in range(len(self.xoffsets)):
            temp_shifted_map = self.stellar_im_vel*self.coupling_maps_shifted[i] # + self.coupling_maps_shifted[i]
            unshifted_temp_map += deshift_coupling_map(temp_shifted_map, self.xoffsets[i], self.yoffsets[i], self.theta)
        
        # PLOT PANEL 2 - shifted, derotated positions
        axs[1].imshow(unshifted_temp_map + self.coupling_map,cmap='RdBu_r', origin='lower', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[1].set_xlabel('X [mas]')
        axs[1].set_title(f'Offset Stellar Positions')
        axs[1].plot(x_new, y_new, marker='x', color='k', alpha=0.1,label='Offset positions')
        # TODO make coupling map contours

        # third panel - plot just one
        temp_shifted_map = self.stellar_im_vel*self.coupling_maps_shifted[i] + self.coupling_maps_shifted[i]
        unshifted_temp_map = deshift_coupling_map(temp_shifted_map, self.xoffsets[i], self.yoffsets[i], self.theta)

        axs[2].imshow(unshifted_temp_map,cmap='RdBu_r', origin='lower', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[2].set_xlabel('X [mas]')
        axs[2].set_title('Last Stellar Position')
        axs[2].plot(x_new, y_new, marker='x', alpha=0.1, color='k', label='Offset positions')

        # make super title
        plt.suptitle(f'Star Coupling Maps, RV aligned\ntheta={self.theta}')
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_star_fiber(self, save_path=None):
        """
        plot the fiber coupling map and overplot the stellar positions with the 
        reference star shown with its velocity map. This shifts the coupling map 
        back to be centered and rotate it so the star is at its correct angle

        input
        -----
        save_path - if not None, will save the figure to the provided path
        """
        fig, axs = plt.subplots(1,3,figsize=[10,4])
        extent = self.plate_scale_coupling * self.coupling_map.shape[0]
        extent = extent.value

        # PLOT PANEL 0 - reference position
        axs[0].imshow((self.stellar_im_vel*self.coupling_map  + self.coupling_map),cmap='RdBu_r', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[0].plot(self.xoffsets, self.yoffsets, marker='x', color='k', label='Offset positions')

        axs[0].set_xlabel('X [mas]')
        axs[0].set_ylabel('Y [mas]')
        axs[0].set_title('Reference')

        def deshift_coupling_map(coupling_map_shifted, xoffset, yoffset, theta):
            """ function to shift the coupling map back to be centered """
            shifted_offset, angle = self._compute_shift(xoffset,yoffset,theta)
            unshifted_map = self._transform_coupling_map(coupling_map_shifted, -shifted_offset, -angle)
            return unshifted_map
        
        # plot de shifted coupling map
        unshifted_temp_map = np.zeros_like(self.coupling_map)
        for i in range(len(self.xoffsets)):
            temp_shifted_map = self.stellar_im_vel*self.coupling_maps_shifted[i] # + self.coupling_maps_shifted[i]
            unshifted_temp_map += deshift_coupling_map(temp_shifted_map, self.xoffsets[i], self.yoffsets[i], self.theta)
        
        # add rotate here
        unshifted_derotated_temp_map = rotate(unshifted_temp_map, self.theta)
        # rotating will create bigger array, so need to crop back down to original size
        midpix = int(unshifted_derotated_temp_map.shape[0] / 2.)
        wz = int(self.coupling_map.shape[0] / 2.)
        unshifted_derotated_map = unshifted_derotated_temp_map[midpix-wz:midpix+wz, midpix-wz:midpix+wz]
        
        # PLOT PANEL 2 - shifted, derotated positions
        axs[1].imshow(unshifted_derotated_map + self.coupling_map,cmap='RdBu_r', origin='lower', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[1].set_xlabel('X [mas]')
        axs[1].set_title('Offset Stellar Positions')
        axs[1].plot(self.xoffsets, self.yoffsets, marker='x', color='k', alpha=0.1,label='Offset positions')
        # TODO make coupling map contours

        # third panel - plot just one
        temp_shifted_map = self.stellar_im_vel*self.coupling_maps_shifted[i] + self.coupling_maps_shifted[i]
        unshifted_temp_map = deshift_coupling_map(temp_shifted_map, self.xoffsets[i], self.yoffsets[i], self.theta)
        test = rotate(unshifted_temp_map, self.theta)
        midpix = int(test.shape[0] / 2.)
        wz = int(self.coupling_map.shape[0] / 2.)
        unshifted_derotated_map = test[midpix-wz:midpix+wz, midpix-wz:midpix+wz]

        axs[2].imshow(unshifted_derotated_map,cmap='RdBu_r', origin='lower', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[2].set_xlabel('X [mas]')
        axs[2].set_title('Last Stellar Position')
        axs[2].plot(self.xoffsets, self.yoffsets, marker='x', color='k', alpha=0.1, label='Offset positions')

        plt.suptitle(f'Star Coupling Maps, Natural View\ntheta={self.theta}')
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()

    def plot_rvs(self, xarr=None, xlabel=None, save_path=None):
        """
        plot the output radial velocities

        inputs
        ------
        xarr - array of x values to plot against, if None will just use the index of the velocities
        save_path - if not None, will save the figure to the provided path
        """
        if xarr is None:
            xarr = np.arange(len(self.vels))

        fig, ax = plt.subplots()
        ax.plot(xarr, self.vels.to(u.km/u.s), marker='o')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel('Measurement Index')
        ax.set_ylabel('Velocity offset [km/s]')
        ax.set_title(f'{self.telescope} DLC Simulation, theta={self.theta}')
        ax.grid()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()

    def plot_lsf_ccf(self,index=0, save_path = None):
        """
        plot the line spread function and ccf for the reference and one of the shifted positions to visualize how the RV shift is happening

        inputs
        ------
        index - which shifted position to plot (default: 0)
        save_path - if not None, will save the figure to the provided path
        """
        # PLot the stellar profile and that convolved with the instrument profile
        import matplotlib.patheffects as pe

        ref_color = 'white'
        shifted_color = 'purple'
        shifted_ccf_color = 'steelblue'

        fig, axs = plt.subplots(2,1,figsize=[6, 5], sharex=True, sharey=True)

        # Plot the stellar velocity profile (vprof)
        axs[0].plot(self.vel_arr/self.vsini, self.vprof_ref, '-', color=ref_color, label='On-axis',lw=5)
        axs[0].plot(self.vel_arr/self.vsini, self.vprofs_shifted[index], '-.', label='Off-axis',lw=5,color=shifted_color)

        # plot the stellar velocity profile convolved with the instrument profile (CCF)
        axs[1].plot(self.vel_arr/self.vsini, self.ccf_ref, '-', label='On-axis CCF',lw=5,color=ref_color)
        axs[1].plot(self.vel_arr/self.vsini, self.ccfs_shifted[index], '-.', label='Off-axis CCF',lw=5,color=shifted_ccf_color)
        
        # plot the bestfit velocity as a vertical line
        vel_ref   = self.vel_arr[np.argmin(self.ccf_ref)] * u.pix
        axs[1].axvline((self.vels[index] + vel_ref)/self.vsini, color=shifted_ccf_color, alpha=0.6)
        axs[1].axvline(self.vel_ref/self.vsini, color=ref_color, alpha=0.6)

        axs[0].set_xlim(-2., 2.)
        axs[0].legend(loc='best',handlelength=2,handletextpad=0.5)#,ncols=2)
        axs[0].set_ylabel('Relative Intensity')
        axs[0].set_title('Stellar Velocity Profile')

        axs[1].set_xlabel('Velocity [fraction of vsini]')
        axs[1].set_title('CCF profile')
        axs[1].set_ylabel('Relative Intensity')
        axs[1].legend(loc='best',handlelength=2,handletextpad=0.5)#,ncols=2)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def printSettings(self):
        """
        print settings for calculation that are listed in dlcSettings dataclass
        """
        print("------CURRENT SETTINGS------")
        attr_to_print = dlcSettings.__dataclass_fields__.keys()
        for attr in attr_to_print:
            print(f"{attr}: {getattr(self, attr)}")
