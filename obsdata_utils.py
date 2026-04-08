# utils to open the excel file and then resave it with information about each observation

import numpy as np
import matplotlib.pylab as plt
import yaml
from astropy.io import fits
import pandas as pd
from astropy.time import Time
import astropy.units as u

from data_utils import get_BC
from astropy.coordinates import SkyCoord

class cObsLog():
    def __init__(self,
                 excel_file: str,
                 data_path: str | None,
                 rv_file: str = '_data/MuGem/MuGem_rv_unbin.csv'):
        
        self.loadObsLog(excel_file)
        
        self.get_gcam_data()
        
        self.get_RV_data(rv_file)

        if data_path is not None:
            self.get_meta_data(data_path)


    def loadObsLog(self,excel_file):
        """
        load observation log 

        inputs
        ------
        excel_file - name and location to the observation log
            first sheet should be the main observing log

        outputs:
        --------
        saves target, timestamp, exptime, xpos, ypos, specfile, gcamfile to class
        """
        self.xl = pd.ExcelFile(excel_file)
        #sheet_names = xl.sheet_names

        # parse main sheet in obs log, will add info to this 
        self.obs_data = self.xl.parse('DLC Targets') # first sheet is the DLC main data log!!!

        # pull out the OG info into variables
        self.target    = self.obs_data['target']
        self.timestamp = self.obs_data['timestamp']
        self.exptime   = self.obs_data['exptime']
        self.xpos      = self.obs_data['xpos']
        self.ypos      = self.obs_data['ypos']
        self.specfile  = self.obs_data['specfile']
        self.gcamfile  = [self.obs_data['gcam1'], self.obs_data['gcam2']]
        self.date      = pd.Series(self.timestamp.astype(str)).str[:8].values
        self.obs_data['date'] = self.date

    def get_gcam_data(self):
        """
        open gcam sheets in excel to merge the measured gcam locations
        """
        # load gcam dta, concatenate all tabs
        self.gcam_data = pd.concat([self.xl.parse('Guider20251017'), 
                                    self.xl.parse('Guider20251016'),
                                    self.xl.parse('Guider20250407'), 
                                    self.xl.parse('Guider20250226')], 
                                    ignore_index=True)

        # go through each gcamfile and fill in positions
        self.gcam_xpos = np.zeros((2,len(self.target))) - 99
        self.gcam_ypos = np.zeros((2,len(self.target))) - 99
        self.strehl = np.zeros((2,len(self.target))) - 99
        self.jitter = np.zeros((2,len(self.target))) - 99
       
        for i,_ in enumerate(self.specfile):
            # cross match with gcam data
            gcam1 = self.gcamfile[0][i]
            gcam2 = self.gcamfile[1][i]

            i1 = np.where(gcam1 == self.gcam_data["#image"])[0]
            i2 = np.where(gcam2 == self.gcam_data["#image"])[0]
            if len(i1)!=0:
                i1 = i1[0]
                if len(i2)==0:
                    self.gcam_xpos[:,i] = [self.gcam_data['xcom'][i1], 0]
                    self.gcam_ypos[:,i] = [self.gcam_data['ycom'][i1], 0]
                    self.strehl[:,i] = [self.gcam_data['strehlproxy'][i1], 0]
                    self.jitter[:,i] = [self.gcam_data['jitterX'][i1], 0]
                else:
                    i2 = i2[0]
                    self.gcam_xpos[:,i] = [self.gcam_data['xcom'][i1], self.gcam_data['xcom'][i2]]
                    self.gcam_ypos[:,i] = [self.gcam_data['ycom'][i1], self.gcam_data['ycom'][i2]]
                    self.strehl[:,i] = [self.gcam_data['strehlproxy'][i1], self.gcam_data['strehlproxy'][i2]]
                    self.jitter[:,i] = [self.gcam_data['jitterX'][i1], self.gcam_data['jitterX'][i2]]
            else:
                pass
        
        self.obs_data['xpos_meas'] = self.gcam_xpos[0]
        self.obs_data['ypos_meas'] = self.gcam_ypos[0]
        self.obs_data['strehl'] = self.strehl[0]
        self.obs_data['jitter'] = self.jitter[0]

    def get_RV_data(self, rv_file='_data/MuGem/MuGem_rv_unbin.csv'):
        """
        open RV data file and merge with obs log
        """
        self.rv_data = pd.read_csv(rv_file)

        # merge with obs log based on matching specfile names, add RVs to obs log
        self.obs_data['serval_RV'] = np.zeros(len(self.obs_data.specfile)) + np.nan
        self.obs_data['serval_RVerr'] = np.zeros(len(self.obs_data.specfile)) + np.nan
        self.obs_data['serval_berv'] = np.zeros(len(self.obs_data.specfile)) + np.nan

        for i,filename in enumerate(self.rv_data['filename']):
            fname = filename.split('/')[-1]
            i_rv = np.where(self.obs_data.specfile == fname)[0]
            if len(i_rv) != 0:
                self.obs_data.loc[i_rv[0], 'serval_RV'] = self.rv_data['rv'].values[i]
                self.obs_data.loc[i_rv[0], 'serval_RVerr'] = self.rv_data['e_rv'].values[i]
                self.obs_data.loc[i_rv[0], 'serval_berv'] = self.rv_data['berv'].values[i]
            else:
                pass
        
    def get_meta_data(self, datapath):
        """
        use barycorrpy to load BC data for each spectrum
        get PA as well
        pull things from file headers like ra, dec, time, etc to add to obs log
        if file not found, set to nan and print warning

        Header keywords to pull:
            P200OBJ = 'USR 0622+2230'      / TCS object name                                
            P200RA  = '06:22:56.99'        / TCS right ascension                            
            P200DEC = '+22:30:52.9'        / TCS declination                                
            P200_UTC= '2025-02-27T02:51:13.971Z' / TCS time                                 
            P200_LST=  / TCS local sidereal time                                            
            P200_HA = ' E00:51:00.0'       / TCS hour angle                                 
            ALTITUDE=              74.2389 / Altitude                                       
            P200_PAR=             -43.6598 / Parallactic angle                              
            AZIMUTH =               130.25 / Azimuthal angle                                
            ZENITH  =              15.7611 / Zenith angle                                   
            P200_AIR=                1.039 / Airmass  
        """
        keys_to_save = ['TIMEWMJD', 'P200OBJ', 'P200RA', 'P200DEC', 'P200_UTC', 'P200_LST', 'P200_HA', 'ALTITUDE', 'P200_PAR', 'AZIMUTH', 'ZENITH', 'P200_AIR', 'INIT_PAR', 'FINL_PAR']
        keys_dtypes = ['str', 'str', 'str', 'str', 'str', 'str', 'str', 'float', 'float', 'float', 'float', 'float', 'float', 'float']
        
        # initiate new columns in obs_data for these header keywords
        for ikey, key in enumerate(keys_to_save):
            self.obs_data[key] = np.zeros(len(self.obs_data.specfile), dtype=keys_dtypes[ikey]) 
            if keys_dtypes[ikey] == 'float':
                self.obs_data[key] -= 9999
        
        bcs = np.zeros(len(self.obs_data.specfile)) - 9999
        weighted_times_bjd = np.zeros(len(self.obs_data.specfile)) - 9999
        pas = np.zeros(len(self.obs_data.specfile)) - 9999

        # step through spec files, load header and save info
        for i,specfile in enumerate(self.obs_data.specfile):
            
            # skip if file doesnt exist
            try:
                hdr = fits.getheader(datapath + specfile)
            except FileNotFoundError:
                print('warning, did not find file %s, setting BC and time to nan for this observation' %(datapath + specfile))
                continue

            # pull header info
            for key in keys_to_save:
                self.obs_data.loc[i, key] = hdr.get(key, np.nan)
            
            # calculate a few things from hdr info
            try:
                coords = SkyCoord(ra=hdr['P200RA'], dec=hdr['P200DEC'], unit=(u.hourangle, u.deg), frame='icrs')
                weighted_times_bjd[i] = Time(hdr['TIMEWMJD'], format='mjd').to_value('jd')
                bcs[i] = get_BC(coords.ra.deg, coords.dec.deg, obstime=hdr['TIMEWMJD'], format='mjd',obsname='Palomar')
                pas[i] = hdr['FINL_PAR'] + hdr['INIT_PAR'] / 2 # average of initial and final parallactic angle
            except Exception as e:
                print('warning, could not calculate BC or PA for file %s, setting to nan for this observation' %(datapath + specfile))
                print(e)
                continue

        # store a few things in self.obs_data
        self.obs_data['bcs'] = bcs
        self.obs_data['parallactic_angle'] = pas
        self.obs_data['obstime_mjd'] = self.obs_data['TIMEWMJD']
        self.obs_data['obstime_bjd'] = weighted_times_bjd
        self.datapath = datapath

    def select_by_target(self,target):
        """
        select observations by target name and store into self.obs_data_subset

        inputs:
        -------
        target - string
            name of the target to select

        outputs:
        --------
        indices of observations matching that target name
        """
        if not hasattr(self, 'obs_data_subset'):
            self.obs_data_subset = self.obs_data.copy()
        
        indices = self.obs_data_subset.target == target
        self.obs_data_subset = self.obs_data_subset[indices]

    def select_by_date(self,
                       date: str,
                       minjd: float | None = None,
                       maxjd: float | None = None):
        """
        select observations by target name

        inputs:
        -------
        date - string
            date to filter out (YYYYMMDD)
        """
        if not hasattr(self, 'obs_data_subset'):
            self.obs_data_subset = self.obs_data.copy()

        indices = self.obs_data_subset.date == date
        if minjd is not None:
            indices &= self.obs_data_subset.obstime_bjd >= minjd
        if maxjd is not None:
            indices &= self.obs_data_subset.obstime_bjd <= maxjd
        
        self.obs_data_subset = self.obs_data_subset[indices]



    def newObsLog(self,newdict,save=False):
        """
        save a new obs log in excel csv format with more columns

        inputs:
        -------
        dictionary of columns to add, length of data must match existing data
        """
        
        self.obs_data_merged = self.obs_data | newdict

        # save this new dic to excel file
        if save: 
            self.obs_data_merged.to_csv('obs_log_merged.csv', index=False)
            print("saved new obs log to obs_log_merged.csv")


    