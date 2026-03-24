# utils to open the excel file and then resave it with information about each observation

import numpy as np
import matplotlib.pylab as plt
import yaml
from astropy.io import fits
import pandas as pd

from data_utils import get_BC

class cObsLog():
    def __init__(self,
                 excel_file: str,
                 data_path: str | None):
        
        self.loadObsLog(excel_file)
        self.getGCamData()
        if data_path is not None:
            self.getBCdata(data_path)

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

    def getGCamData(self):
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

    def getBCdata(self, datapath):
        """
        use barycorrpy to load BC data for each spectrum
        """
        bcs = np.zeros(len(self.obs_data.specfile))
        weighted_times = np.zeros(len(self.obs_data.specfile))
        for i,specfile in enumerate(self.obs_data.specfile):
            try:
                hdr = fits.getheader(datapath + specfile)
                bcs[i] = get_BC(source_name=self.target[i], obstime=hdr['TIMEWMJD'], format='mjd',obsname='Palomar')
                weighted_times[i] = hdr['TIMEWMJD']
            except FileNotFoundError:
                pass

        self.obs_data['bcs'] = bcs
        self.obs_data['weighted_times'] = weighted_times
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
                       date: str):
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


    