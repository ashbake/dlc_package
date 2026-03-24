from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from scipy.ndimage import zoom, shift
from pathlib import Path

from astropy.convolution import Gaussian1DKernel, convolve
from flatstar import draw
import yaml

from astropy import units as u
from astropy import constants as c


from dataclasses import dataclass, field

FILEPATH = Path(__file__).resolve().parent

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
                               StellarModel):
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

        if config != None: 
            self._loadConfig(config) # override deafult settings with config file if provided
        
        self.diagnostics_on = diagnostics_on 

        # run through presets to load necessary files and settings for the class
        self.loadCouplingMap()

    def loadCouplingMap(self):
        """ load coupling map for the telescope and zoom in on it

            loads plate scale for the coupling map
        """
        if self.telescope=='Keck':
            coupling_file_name = 'staticModel_Keck_HK_defoc0nmRMS_LO0nmRMS_Fnum2.63_atm0_adc0_nWvls10_PL0.fits'
        #elif telescope=='Hale':
        #    coupling_map = ""
        elif self.telescope == 'TMT' or self.telescope == 'Thirty Meter Telescope':
            coupling_file_name = 'staticModel_TMT_HK_defoc0nmRMS_LO0nmRMS_Fnum2.61_atm0_adc0_nWvls10_PL0.fits'
        else:
            raise(Warning, 'Telescope Data not available, options are: Keck, TMT')
        
        coupling_map_path = FILEPATH / "data/coupling_maps/" / coupling_file_name
        print(f'Loading {coupling_map_path}')
        map_ext = 7 # coupling map extension (wavelength)

        # make wavelength array for each layer of coupling map cube
        wstart = fits.getval(coupling_map_path, 'LAMMIN')
        wend = fits.getval(coupling_map_path, 'LAMMAX')
        nwvls = fits.getval(coupling_map_path, 'NAXIS3')
        warr_fits = np.linspace(wstart, wend, nwvls)

        self.plate_scale_coupling_raw = fits.getval(coupling_map_path, 'DXMAS') * u.mas / u.pixel# mas per pixel

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
            plt.imshow(self.coupling_map)
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
                       x: u.mas,
                       y: u.mas,
                       theta: u.deg):
        """ translate the shift in x,y to angular deg to apply to coupling map
        in a way that utilizes the fiber symmetry
 
        inputs
        ------
        x,y - offset position of star w.r.t. fiber (u.mas)
        theta - angle of stellar axis w.r.t. xy plane (u.deg)

        returns
        -------
        shifted_offset - offset collapsed to 1D (u.mas)
        angle          - angle of shift to apply to coupling map (u.deg)
        ref_angle      - angle of star in 1D axis (u.deg)
        """
        shifted_offset   = np.sqrt(x**2 + y**2)  # mas, from center of field
        if x==0: 
            phi=90 * u.deg
        else:
            phi = np.arctan(y/x) # bc x and y have units, returns radians already

        angle            = phi + theta           # degrees, shifts coupling map
        ref_angle        = theta                 # this reference angle will differ now

        return shifted_offset, angle.to(u.deg)
    
    def _apply_shift(self, coupling_map, shifted_offset, angle):
        """ apply the shift to the coupling map based on the shifted offset and angle
        """
        # compute the desired shift in image pixel units
        shifted_offset_pix = shifted_offset  * 1./ self.plate_scale_coupling   # pixels

        # shift coupling map accordingly
        coupling_map_shifted = shift(coupling_map, [shifted_offset_pix.decompose().value * np.sin(angle), 
                                                    shifted_offset_pix.decompose().value * np.cos(angle)],mode='wrap')
        
        # make it so coupling maps have same peak value (which gets messed up when shifting)
        unshifted_sum_coupling = np.max(self.coupling_map)
        coupling_map_shifted *= (unshifted_sum_coupling / np.max(coupling_map_shifted))

        return coupling_map_shifted
    
    def _create_star_image(self, ):
        """
         create a limb darkened star image with velocity map based on vsini and diameter of star
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
        
        return im_intensity, vel_arr, stellar_im_vel, vel_scale_pix
    
    @u.quantity_input
    def _create_lsfs(self, 
                           shifted_offset: u.mas, 
                           angle: u.deg):
        """
        TODO - finish this docstring and converting function to class
        """
        # Shift Coupling Map
        coupling_map_shifted   = self._apply_shift(self.coupling_map, shifted_offset, angle)

        # Create stellar image intensity and respective velocity map
        im_intensity, vel_arr, stellar_im_vel, vel_scale_pix = self._create_star_image()

        # Create Integrated Line Profiles
        scale_fac_ref      = np.nanmax(np.nansum(im_intensity * self.coupling_map, axis=0))
        scale_fac_shift    = np.nanmax(np.nansum(im_intensity * coupling_map_shifted, axis=0))

        # Integrate the line profile along for reference
        vprof = np.nansum(im_intensity * self.coupling_map, axis=0)
        vprof /= scale_fac_ref
        vprof = (1. - vprof)

        # *** do the scalings serve to keep the total flux consistent (shifted flux will be less bc off fiber?)
        vprof_shifted = np.nansum(im_intensity * coupling_map_shifted, axis=0)
        vprof_shifted /= scale_fac_shift
        vprof_shifted = (1. - vprof_shifted)

        # section for generating the spectrograph PSF convolution kernel
        spec_res_vel =  c.c / self.R # spectrometer resolution 
        spec_kernel_wid = spec_res_vel / vel_scale_pix # PSF FWHM in image pixels
        spec_kernel_sigma = spec_kernel_wid / 2.335 # sigma of gaussian in pixels
        spec_kernel = Gaussian1DKernel(spec_kernel_sigma.decompose().value).array # final kernel
        print('res vel', spec_res_vel, 'spec kernel wid', spec_kernel_wid)

        # convolve the integrated line profil with the fake spectrometer PSF kernel
        ccf_ref  = convolve(vprof,spec_kernel,normalize_kernel=True, boundary='extend',fill_value=0)
        ccf_shifted  = convolve(vprof_shifted,spec_kernel,normalize_kernel=True, boundary='extend',fill_value=0)

        return coupling_map_shifted, im_intensity, stellar_im_vel, vel_arr, vprof, vprof_shifted, ccf_ref, ccf_shifted

    def plot_star_fiber(self):
        """
        plot the fiber coupling map and overplot the stellar positions with the reference star shown with its velocity map
        """
        fig, axs = plt.subplots(1,3,figsize=[10,4])
        extent = self.plate_scale_coupling * len(self.stellar_im_vel)
        extent = extent.value
    
        axs[0].imshow((self.stellar_im_vel*self.coupling_map  + self.coupling_map),cmap='RdBu_r', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[1].imshow((self.stellar_im_vel*self.coupling_maps_shifted[0]  + self.coupling_maps_shifted[0]),cmap='RdBu_r', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[2].imshow((self.stellar_im_vel*self.coupling_maps_shifted[len(self.xoffsets)-1]  + self.coupling_maps_shifted[len(self.xoffsets)-1]),cmap='RdBu_r', extent=(-extent/2, extent/2, -extent/2, extent/2))

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

        plt.show()

    def plot_star_fiber_rotated(self):
        """
        plot the fiber coupling map and overplot the stellar positions with the reference star shown with its velocity map
        shift the coupling map back to be centered and rotate it so the star is at its correct angle
        """
        fig, axs = plt.subplots(1,3,figsize=[10,4])
        extent = self.plate_scale_coupling * len(self.stellar_im_vel)
        extent = extent.value
        
        # attempt to undo shift to coupling map star combo
        first_shifted_map = self.stellar_im_vel*self.coupling_maps_shifted[0]  + self.coupling_maps_shifted[0]
        shifted_offset, angle = self._compute_shift(self.xoffsets[0], self.yoffsets[0], self.theta)
        unshifted_first_map = self._apply_shift(first_shifted_map, -1*shifted_offset, angle)

        # attempt to undo shift to coupling map star combo
        last_shifted_map = self.stellar_im_vel*self.coupling_maps_shifted[len(self.xoffsets)-1]  + self.coupling_maps_shifted[len(self.xoffsets)-1]
        shifted_offset, angle = self._compute_shift(self.xoffsets[-1], self.yoffsets[-1], self.theta)
        unshifted_last_map = self._apply_shift(last_shifted_map, -1*shifted_offset, angle)

        axs[0].imshow((self.stellar_im_vel*self.coupling_map  + self.coupling_map),cmap='RdBu_r', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[1].imshow(unshifted_first_map,cmap='RdBu_r', extent=(-extent/2, extent/2, -extent/2, extent/2))
        axs[2].imshow(unshifted_last_map,cmap='RdBu_r', extent=(-extent/2, extent/2, -extent/2, extent/2))

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

        plt.show()



    def plot_rvs(self, ):
        """
        plot the output radial velocities
        """
        fig, ax = plt.subplots()
        ax.plot(self.yoffsets, self.vels.to(u.km/u.s), marker='o')
        ax.set_xlabel('Position [mas]')
        ax.set_ylabel('Velocity offset [km/s]')
        ax.set_title(f'{self.telescope} DLC Simulation, theta={self.theta}')
        ax.grid()
        plt.show()


    def run(self):
        """ 
        run the DLC simulation with the current settings in the class
        """
        xoffset = self.xoffsets[0]
        
        ccfs_shifted = {}
        coupling_maps_shifted = {}
        vels = np.zeros(len(self.yoffsets)) * u.km / u.s / u.pix
        for i,yoffset in enumerate(self.yoffsets):
            shifted_offset, angle = self._compute_shift(xoffset,yoffset,self.theta)
            print(yoffset)
            print(angle)
            print(shifted_offset)
            coupling_map_shifted, im_intensity, stellar_im_vel,\
                vel_arr, vprof, vprof_shifted, ccf_ref, ccf_shifted\
                    = self._create_lsfs(shifted_offset, angle)
            ccfs_shifted[i] = ccf_shifted
            coupling_maps_shifted[i] = coupling_map_shifted
            print('ccf min', np.argmin(ccf_shifted))
            vel_ref   = vel_arr[np.argmin(ccf_ref)]
            vel_shift = vel_arr[np.argmin(ccf_shifted)]
            print('vel', vel_shift - vel_ref)
            vels[i]   = vel_shift - vel_ref
            # save images of positions
            #plt.figure()
            #plt.imshow(coupling_map_shifted * im_intensity + coupling_map_shifted)
            #middle_of_map = np.where(coupling_map_shifted == np.max(coupling_map_shifted))
            #plt.plot(middle_of_map[1],middle_of_map[0],'kx')
            #plt.savefig('./yposition_%smas_xposition_%smas_angle_%sdeg.png'%(yoffset, xoffset,self.theta))
        
        self.vels = vels * u.pix # don't't need the pixels unit

        # store other things for just the last iter for now for plotting
        self.stellar_im_vel = stellar_im_vel
        self.im_intensity = im_intensity
        self.coupling_maps_shifted = coupling_maps_shifted
        self.ccfs_shifted = ccfs_shifted

    def printSettings(self):
        """
        print settings for calculation that are listed in dlcSettings dataclass
        """
        print("------CURRENT SETTINGS------")
        attr_to_print = dlcSettings.__dataclass_fields__.keys()
        for attr in attr_to_print:
            print(f"{attr}: {getattr(self, attr)}")
