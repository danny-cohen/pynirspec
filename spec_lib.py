#!/usr/bin/env python
# coding: utf-8
""" Functions for reducing and processing echelle spectra from NIRSPEC (or NIRSPAO). 
    
"""
import matplotlib, pylab, os, astropy, copy
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings
warnings.filterwarnings(action='ignore')
import numpy.ma as ma
import numpy.fft as fft
from scipy import ndimage as nd
from scipy import interpolate
import scipy.optimize as opt
import scipy.signal as signal
from scipy.signal import argrelextrema,find_peaks,windows

# astropy:
from astropy.io import ascii,fits
import astropy.wcs 
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from astropy.constants import c
from astropy.stats import mad_std,sigma_clip,sigma_clipped_stats
from astropy.time import Time
from astropy.convolution import Gaussian1DKernel,Box1DKernel,Gaussian2DKernel,Box2DKernel,interpolate_replace_nans,convolve
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord,EarthLocation
from astropy.table import Table,QTable
from astropy.visualization import ImageNormalize, LogStretch, AsinhStretch, SqrtStretch, MinMaxInterval, ZScaleInterval
# astropy-affil 
from regions import read_ds9,write_ds9,PixCoord,RectanglePixelRegion
from specutils import Spectrum1D, SpectralRegion, SpectrumCollection
from specutils.manipulation import noise_region_uncertainty,snr_threshold,\
        FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler,\
median_smooth, box_smooth, gaussian_smooth, trapezoid_smooth
from astroscrappy import detect_cosmics
import ccdproc as ccdp
from spectral_cube import SpectralCube
from reproject.mosaicking import find_optimal_celestial_wcs,reproject_and_coadd
from reproject import reproject_interp,reproject_exact,reproject_adaptive

import fitting


# clean hot/cold pixels from data using astroscrappy.detect_cosmics
def cleancosmic(data,maxiter=4,sigclip=4.5,sigfrac=0.2,objlim=5.0,mask=None,satlvl=65000.,gain=5.0,rdnoise=10.8):
    """
    cleaned_data, bad_mask = cleancosmic(data,mask=None,...) where ... are optional arguments for detect_cosmics
    
    """
    crmask,cleandata=detect_cosmics(data, inmask=mask, sigclip=sigclip, sigfrac=sigfrac, objlim=objlim, gain=gain, \
    readnoise=rdnoise, satlevel=satlvl, niter=maxiter, sepmed=True, cleantype='medmask', fsmode='median') 
    
    return cleandata, crmask


def noiseimg(objdata, flatdata, texp, gain=5.0, rdnoise=10.8, darkcur=0.67 ):
    """
    objdata should be rectified image in ADU, including object + sky
    flatdata should be normalized, rectified flat-field
    texp is total exposure time to measure dark current in e- -> ADU
    
    gain in e-/ADU
    rdnoise should be in e-
    darkcur in e-/s
    """    
    # read out noise should be squared
    ronoise=np.square(rdnoise/gain) # readout noise variance in ADU 
    dcnoise=darkcur*texp/gain # dark current noise in ADU
    
    # multiply by gain to get image in electrons
    var=np.abs( objdata - np.sqrt(2.)*np.sqrt(ronoise)) + ronoise + dcnoise # in electron^2
    
    # divide by normalized flat squared
    var/=np.square(flatdata)
    
    # RETURN VARIANCE IMAGE
    return var


def fitbkg(data,order=[2,2],fitsec=() ):
    """Fit and subtract a polynomial surface to a 2D image. 
    Parameters
    ----------
        data: str or list of str
            Raw flat-field image(s) to be reduced.
        fitsec: tuple of 4-element lists
            Sections of image to include in fit. Each section should be an element in the tuple giving [x1,x2,y1,y2].
            Multiple sections are entered as separate tuple elemnts.
        offimage: str or list of str, default: None
    """
    xx,yy = np.meshgrid(np.arange(data.shape[1]),np.arange(data.shape[0]))#[:data.shape[0], :data.shape[1]]
    p_init = models.Legendre2D(order[0],order[1])
    fit_p = fitting.LinearLSQFitter()#LevMarLSQFitter()
    
    # make mask for ipxels to include
    fitmask=np.ones(data.shape)
    nsec=len(fitsec)
    if nsec>0:
        for sec in fitsec:
            x1,x2,y1,y2=sec
            fitmask[y1:y2,x1:x2]=0.0
    fitmask=fitmask.astype(np.bool)
    ifit=np.where(fitmask==False)
    #print(np.size(ifit),np.size(xx[ifit]),np.size(xx))
    p = fit_p(p_init, xx[ifit], yy[ifit], data[ifit])
    model=p(xx,yy)
    residual=data-model
    
    return residual, model, fitmask


def heliocorr(targ, obstime, observatory='Keck'):
    
    #obstime = Time('2017-2-14')
    if type(targ)==str: # then assume to be the target's name
        target = SkyCoord.from_name(targ)  
    elif type(targ)==astropy.coordinates.sky_coordinate.SkyCoord:
        target=targ
        
    loc = EarthLocation.of_site(observatory)  
    corr = target.radial_velocity_correction('heliocentric',obstime=obstime, location=loc).to('km/s')  
    
    return corr
    
def wave2vel(wave,waverest=4.052262*u.um,kind='optical'):
    
    if kind=='optical':
        
        vel=c.to(u.km/u.s) * (wave.to(u.um) - waverest.to(u.um)) / waverest.to(u.um)
        
    elif kind=='radio':
        
        vel=c.to(u.km/u.s) * (wave.to(u.um) - waverest.to(u.um)) / wave.to(u.um)
        
    return vel

def vel2wave(vel,waverest=4.052262*u.um,kind='optical'):
    
    if kind=='optical':
        
      #  vel=c.to(u.km/u.s) * (wave.to(u.um) - waverest.to(u.um)) / waverest.to(u.um)
        
        wave=  waverest.to(u.um) * (1 + (vel.to('km/s') / c.to('km/s')))
        
    elif kind=='radio':
        
#         vel=c.to(u.km/u.s) * (wave.to(u.um) - waverest.to(u.um)) / wave.to(u.um)
        
        wave= waverest / (1 - (vel.to('km/s') / c.to('km/s')))
        
    return wave


def spatrect(data,spatfile):

    """ 
        Perform spatial rectification on NIRSPEC spectrum.
    
        data -- should be 2D numpy array, i.e. the un-rectified spectrum read into python with astropy.io
        spatfile -- Spatmap file, e.g. 'spat.map'
    
    """
    
   # if rotate:
   #     data=np.rot90(data,-1)
    
    # Read parameters and spatial map.
    fo=open(spatfile,'r')
    recfitorder,xdim,clip_ya,clip_yb,wavecntr,wavedelta=fo.readline().split()
    fo.close()
    recfitorder=int(recfitorder)
    xdim=int(xdim)
    clip_ya=int(clip_ya)
    clip_yb=int(clip_yb)
 #   dataclip=data[clip_ya:clip_yb+1,:]
    wavecntr=float(wavecntr)
    wavedelta=float(wavedelta)
    ydim=clip_yb-clip_ya+1
    wavemin=wavecntr-(ydim/2.)*wavedelta
    waveout=wavemin+wavedelta*(np.arange(ydim)+0.5)
    # get input spatial map and clipped data
    spatmap=np.loadtxt(spatfile,skiprows=1,usecols=range(1,recfitorder+2))
    datain=data[clip_ya:clip_yb+1,:]
    # define input grids and empty arrays
    xx,yy=np.meshgrid(np.arange(xdim),np.arange(ydim))
    xxb,yyb=np.meshgrid(np.arange(xdim),np.arange(ydim+1)-0.5)
    rectified=np.zeros(datain.shape)
    waveorig=np.zeros(datain.shape)
    for i in range(xdim):
        wavein=np.zeros(ydim)
        waveinb=np.zeros(ydim+1)
        for k in np.arange(0,recfitorder+1):
            wavein+=spatmap[i,k]*yy[:,i]**k
            waveinb+=spatmap[i,k]*yyb[:,i]**k
        dout_din=wavedelta/(waveinb[1:]-waveinb[:-1])
        interpolator=interpolate.interp1d(wavein,datain[:,i],kind='quadratic',\
        fill_value='extrapolate',assume_sorted=True)
        rectified[:,i]=interpolator(waveout)*dout_din
        waveorig[:,i]=wavein


    return rectified,waveout,waveorig


def specrect(data,specfile):

    """ 
        Perform spatial rectification on NIRSPEC spectrum.
    
        data -- should be 2D numpy array, i.e. the un-rectified spectrum read into python with astropy.io
        spatfile -- Spatmap file, e.g. 'spat.map'
    
    """
    # Read parameters and spectral map.
    fo=open(specfile,'r')
    recfitorder,xdim,clip_ya,clip_yb,wavecntr,wavedelta=fo.readline().split()
    fo.close()
    recfitorder=int(recfitorder)
    xdim=int(xdim)
    clip_ya=int(clip_ya)
    clip_yb=int(clip_yb)
 #   dataclip=data[clip_ya:clip_yb+1,:]
    wavecntr=float(wavecntr)
    wavedelta=float(wavedelta)
    ydim=clip_yb-clip_ya+1
    wavemin=wavecntr-(xdim/2.)*wavedelta
    waveout=wavemin+wavedelta*(np.arange(xdim)+0.5)
    specmap=np.loadtxt(specfile,skiprows=1,usecols=range(1,recfitorder+2))
    datain=data.copy()

    # define input grids and empty arrays
    xx,yy=np.meshgrid(np.arange(xdim),np.arange(ydim))
    xxb,yyb=np.meshgrid(np.arange(xdim+1)-0.5,np.arange(ydim))
    rectified=np.zeros(datain.shape)
    waveorig=np.zeros(datain.shape)
    for j in range(ydim):
        wavein=np.zeros(xdim)
        waveinb=np.zeros(xdim+1)
    
        for k in np.arange(0,recfitorder+1):
            wavein+=specmap[j,k]*xx[j,:]**k
            waveinb+=specmap[j,k]*xxb[j,:]**k
    
        dout_din=wavedelta/(waveinb[1:]-waveinb[:-1])
        interpolator=interpolate.interp1d(wavein,datain[j,:],kind='quadratic',\
        fill_value='extrapolate',assume_sorted=True)
        rectified[j,:]=interpolator(waveout)*dout_din
        waveorig[j,:]=wavein

    return rectified,waveout,waveorig
        

def rectify(data,spatmap,specmap,return_full=False):
    """
    Use spatrect and specrect to fully rectify 2D echelle spectra. Requires input spatial and spectral maps produced
    with SPATMAP.PRO and SPECMAP.PRO in the UCLA REDSPEC library
    """

    spat,spatgrid,spatorig=spatrect(data,spatmap)
    rectified,wavegrid,waveorig=specrect(spat,specmap)
    
    if return_full:
        return rectified,wavegrid,spatgrid,waveorig,spatorig
    else:
        return rectified,wavegrid,spatgrid
    
    


def fringe(y,freq_bounds=[],medwidth=51,interactive=True):
    
    """ 
        Remove fringe patterns in 1D spectrum using Hanning filter in FFT domain.
        Subtract a smoothed spectrum (median filter width = filtwidth), compute FFT,
        and remove the freqs in power spectrum corresponding to fringing.
    
    """
    
    #~~ name parameters that don't change throughout procedure
    xdim = y.size
    xpix=np.arange(xdim)+1
    #print("X-dim: ",xpix.size)
    xdimfft=250
    nfilt=xdim//2+1
    freq=np.arange(nfilt)
  #  print(nfil)
    specorig = copy.copy(y)
    specmed = np.median(specorig)
    specsmooth = signal.medfilt(specorig,medwidth) # yields different result than IDL median
    specfft = fft.fft(specorig-specsmooth)#specsmooth)
    yp = np.abs( specfft[:nfilt] )**2
    yp /=np.max(yp)

        # yprms = np.std(yp)
        # ipeaks,prop=signal.find_peaks(yp/yprms,height=thresh_peak)
        # widths=signal.peak_widths(yp/yprms,ipeaks,rel_height=0.8)
        # fpeaks=freq[2:nfilt][ipeaks]
        # flo=fpeaks-widths[0]/2.
        # fhi=fpeaks+widths[0]/2.
        # heights=prop['peak_heights']

    if interactive:
        fig = plt.figure()
        ax1=fig.add_subplot(111)
        #ax2 = fig.add_subplot(212)
        # subtract smooth background and compute FFT
       # freq=np.arange(nfil,dtype='float')
        yp = np.abs( specfft[0:nfilt] )**2
        ypmax = np.max(yp)
        yp/=ypmax

        # plot power spectrum and intensity
        ax1.plot(freq,yp,'k-',lw=2)
        ax1.set_xlim(0,xdimfft)
        ax1.set_xlabel("Frequency [1/%i pix]"%xdim)
        ax1.set_ylabel("Power")
        ax1.annotate("Power Spectrum",xy=(0.75,0.75),xycoords=('axes fraction'))
        #ax1.set_xticks(freq[::25])
        pltymax=np.max( (specorig/specmed)[100:-100] )+0.2 
        pltymin =np.min( (specorig/specmed)[100:-100] )-0.2
        # ax2.set_ylim(pltymin,pltymax)
        # ax2.set_xlim(1,xdim)
        # ax2.set_xlabel("Pixel")
        # ax2.set_ylabel("Intensity")
        # ax2.annotate("Intensity Spectrum",xy=(0.75,0.75),xycoords=('axes fraction'))
        # ax2.set_title("Intensity Spectrum")
        plt.show(block=False)
        if np.size(freq_bounds)>=2:
            freq_low=[x[0] for x in freq_bounds]
            freq_hi=[x[1] for x in freq_bounds]
        elif np.size(freq_bounds)<2:
        # Allow User to pick lines if not already supplied
            freq_low=[]
            freq_hi=[]
            lines=""
            while True:   
                lines=input("Enter low and high frequency bounds for range to remove, hit return after each window: `flo1,fhi1': ")
                if lines=='q':
                    break
                else:
                    freq_low.append(float(lines.split(',')[0]))
                    freq_hi.append(float(lines.split(',')[1]))
                    ax1.vlines([float(lines.split(',')[0]),float(lines.split(',')[1])],0.,1.0,linestyle='--')
        #plt.show(block=False)
    else:
        freq_low=[x[0] for x in freq_bounds]
        freq_hi=[x[1] for x in freq_bounds]
        
    freq=np.arange(nfilt//2+1,dtype='float') / (nfilt / float(xdim) )
    nwindow=len(freq_low)
    #print("Filter windows: ",len(freq_low))
    fil = np.ones(freq.size)
    for w in range(nwindow):
        fil *= ( (freq > freq_hi[w]) | (freq < freq_low[w]) ).astype('float')
        
    # Now actually perform the filtering to remove fringing
    fil = np.concatenate((fil,np.fliplr([fil[1:]])[0]))
    fil = np.real(fft.ifft(fil)) # don't need the next step from REDSPEC (fil=fil/nfil)
    fil = np.roll(fil,nfilt//2)
    fil = fil*np.hanning(nfilt) # basically same as in IDL though very subtle differences in edge values
    # convolve spectrum with hanning filter window
    ycorr = nd.filters.convolve(specorig-specsmooth, fil, mode='wrap') + specsmooth
    out_bounds=[(l,h) for l,h in zip(freq_low,freq_hi)]

    return ycorr,out_bounds


    
#~~
def fringecor(data,medwidth=51,fringe_row=(0,-1),freq_bounds=[]):

    """
        Remove fringe patterns in 1D spectrum using Hanning filter in FFT domain.
        Subtract a smoothed spectrum (median filter width = filtwidth), compute FFT,
        and remove the freqs in power spectrum corresponding to fringing.

    """

    #~~ name parameters that don't change throughout procedure
    xdim = data.shape[1]
    ydim = data.shape[0]
    
    if len(freq_bounds)==0:
        
        yinit=np.median(data.copy()[fringe_row[0]:fringe_row[1],:],0)
        yinc,frq_clip=fringe(yinit,medwidth=medwidth,interactive=True)
   # print(yinc,frq)
    elif len(freq_bounds) >=1:
        frq_clip=copy.copy(freq_bounds)
    
    # loop through rows of echellogram and apply fringe correction to each on
    datacorr=np.zeros(data.shape)
    for j in range(ydim):
        datacorr[j,:],_=fringe(data.copy()[j,:],freq_bounds=frq_clip,\
        medwidth=medwidth,interactive=False)

    return datacorr
    

    
def normflat(data, rectmap, habounds=(0,2000), lthresh=500., badval = 1.):
    """ Normalize flat-field image (echelle order specified in input `rectmap`). """
    # make copy of data
    datacopy=data.copy()
    # make coordinate grids
    ndim=data.shape
    x, y = np.meshgrid(np.arange(ndim[0]),np.arange(ndim[1]))
    #print(x)
    # read spectral map and get boundaries of the spectral order
    fo=open(rectmap,'r')
    lim=fo.readline().split()[2:4]
    fo.close()
    vabounds=(int(lim[0]), int(lim[1]))
    # ignore pixels beyond column 1000 by setting value to 1.0
    mask = (y < vabounds[0]) | (y > vabounds[1]) | (x <= habounds[0]) | (x> habounds[1]) | (datacopy < lthresh)
    # take mean of the non-masked pixels
    mean = ma.mean(ma.masked_array(datacopy, mask=mask))
    # create normalized data array
    normalized = datacopy * (1-mask.astype(int)) / mean #/ mean
    # # set masked pixels to 1
    # normalized[np.where(mask==True)]=1.0
    # also mask where pixels blow up when div by mean, near edges
    normalized[np.where(normalized > 2.0)] = 1.0
    normalized[np.where(normalized < 0.5)] = 1.0
    # avoid zeroes (not to sure about these)
    normalized[np.where(normalized == 0.0)] = 1.0    
    
    masknorm = normalized==1.0
    
    newmean=ma.masked_array(datacopy,mask=masknorm).mean()
    
    newdata= datacopy / newmean
    
    newdata[np.where(masknorm==True)]=1.0
    
    # do one more iteration? 
    return newdata



#~~ Generate normalized, master flat-field spectrum from raw image(s).
def redflat(flatimage,outimage=None,normalize=True,rectmap='spec.map',\
            norm_xrange= (5,2000), norm_thresh = 1000., clean=True, darkimage=None):
    """ Produce reduced (master) flat-field spectrum. 
    
    Parameters
    ----------
        flatimage : str or list of str
            Raw flat-field image(s) to be reduced.
        outimage: str 
            Filename for output reduced flat-field image.
        normalize: bool, optional, default: True
            If False, the flat will not be mean-normalized and off-order pixels will not be replaced by value 1.
        rectmap: str, optional, default: 'spec.map'
            File used for order normalization (no rectification is performed). Only used if normalize is True. 
        norm_xrange: 2-element tuple of ints, default: (5, 2000)
            Image pixel boundaries along dispersion direction, typically columns, containing the region used in normalization.
        norm_thresh: float, optional, default: 1000.
            Lowest pixel value used in normalization. Should distinguish between off-order background.
        clean: bool, optional, default: True
            If False, will not perform cleaning of bad pixels/cosmic rays.
        darkimage: str or list of str, optional, default: None
            Dark image(s) to be subtracted from the flat. Should have identical exposure time as the flat. If a list, images will
            be combined prior to subtraction. 
        
    Returns
    -------
        hdu: astropy.io.fits.PrimaryHDU
            Reduced master flat-field image. Save with hdu.writeto(filename)     
    """
    # Make combined flat
    if np.size(flatimage)>1:
        hdrout=fits.getheader(flatimage[0]).copy()
        flats=[] # to be list of reduced flat images
        itime=[]
        print("Reducing and combining flat frames...")
        for f in flatimage:
            # read data
            hdu=fits.open(f)[0]
            dat=hdu.data
            hdr=hdu.header
            itime.append(hdr['elaptime'])
            # clean bad pixels
            if clean:
                print("Cleaning bad pixels from frame")#" %s"%hdr['filename'])
                specfile,bpm=cleancosmic(dat.copy(),sigfrac=0.5)
               # datcopy
            else:
                specfile=dat.copy()
                bpm=np.zeros(specfile.shape)
            specfile[np.where(bpm==True)]=np.nan
            kern=Box2DKernel(6)
            specfile=interpolate_replace_nans(specfile,kern)  
            flats.append(specfile)
        # combine, masking bad pixels
        flatout=np.median(flats,axis=0)
        flatitime=np.mean(itime)
    else:
        hdu=fits.open(flatimage)[0]
        dat=hdu.data
        hdrout=hdu.header.copy()
        flatitime=hduout.header['elaptime']
        if clean:
            flatout,bpm=cleancosmic(dat.copy(),sigfrac=0.5)
            flatout[np.where(bpm==True)]=np.nan
            kern=Box2DKernel(6)
            flatout=interpolate_replace_nans(flatout,kern)  
        else:
            flatout=dat.copy()
     
    # now do dark reduction and subtraction if a list of dark image files is supplied
    if type(darkimage)!=type(None):    
        if np.size(darkimage)>1:
            hdrdark=fits.getheader(darkimage[0]).copy()
            # Make combined dark frame for flat
            darks=[] # to be list of reduced dark images
           # itime=[]
            print("Cleaning and combining dark frames...")
            for f in darkimage:
                # read data
                hdu=fits.open(f)[0]
                dat=hdu.data
                hdr=hdu.header
              #  itime.append(hdr['elaptime'])
        
                # clean bad pixels
                if clean:
                    print("Cleaning bad pixels from frame")#": %s"%hdr['filename'])
                    specfile,bpm=cleancosmic(dat.copy())
                else:
                    specfile=dat.copy()
                    bpm=np.zeros(dat.copy().shape)
                specfile[np.where(bpm==True)]=np.nan
                kern=Box2DKernel(6)
                specfile=interpolate_replace_nans(specfile,kern)
                # append reduced data to list of reduced flats
                darks.append(specfile)
            # combine and save fits image
            darkout=np.mean(darks,axis=0) 
            #darkitime=np.mean(darkitime)
        else:
            hdu=fits.open(darkimage)[0]
            dat=hdu.data
            hdrdark=hdu.header.copy()
            #darkitime=hdrdark['elaptime']
            if clean:
                darkout,bpm=cleancosmic(dat.copy(),sigfrac=0.5)
                darkout[np.where(bpm==True)]=np.nan
                kern=Box2DKernel(6)
                darkout=interpolate_replace_nans(darkout,kern)  
            else:
                darkout=dat.copy()
                
        # DARK SUBTRACTION    
        flatout=flatout-darkout
    # end dark reduction
    
    # NORMALIZATION
    if normalize:
        flatout=normflat(flatout,rectmap,lthresh=norm_thresh,habounds=norm_xrange)
    
    # SAVE IF OUTPUT!=None
    if type(outimage)==str:
        fits.writeto(outimage,data=flatout,header=hdrout,overwrite=True,output_verify='ignore')
    
    return flatout
    

# FULL REDUCTION OF RAW NIRSPEC/NIRSPAO SPECTRA CONTAINED IN REDSPEC FUNCTION
def redspec(image, outimage, offimage=None, flatimage=None, mode=0,spatmap='spat.map',specmap='spec.map', \
            clean=True , trim_bounds=[0,1000,8,156], bkg_subtract=False, bkg_order=1, bkg_sec=([0,250,0,-1],[500,-40,0,-1]), \
            fringe_corr=False, fringe_width=51, fringe_freq=[], exptimekey='elaptime', \
            restwav=4.052262, gain=5.0, rdnoise=10.0, darkcur=0.67, target='' ):
    """ REDUCE SPEC DATA FROM NIRSPEC (+NIRSPAO). 
    
    Parameters
    ----------
        image : str or list of str
            Raw flat-field image(s) to be reduced.
        outimage: str 
            Filename for output reduced flat-field image.
        offimage: str or list of str, default: None
            Raw off-frame/sky image(s) to be subtracted from the input image. Also valid for B frame in A-B nod pairs.
        flatimage: str or list of str, default: None
            Master normalized flat-field image(s). If None, no flattening performed.           
        mode: int (0 or 1), default: 0
            Mode of reduction specifying order or operations. If 1: rectification before reduction. If mode=0 or not 1: reduction first.
        spatmap: str , default: `spat.map`
            Spatial rectification file.
        specmap: str , optional, default: `spec.map`
            Spectral rectification file.
        clean: bool, optional, default: True
            If False, will not perform cleaning of bad pixels/cosmic rays.
        bkg_subtract: bool, optional, default: False
            If True, subtract a background spectrum from each row. The background spectrum is extracted within a box with spatial boundaries defined by bkg_box_range.
        bkg_box_range: tuple or list of ints
            The boundaries along the spatial axis of the region used to calculate background (pixel coordinates)
        fringe_corr: bool, optional, default: False
            If True, perform do interactive fringe-correction, applying to all rows of the spectrum. 
        fringe_width: int (odd), optional, default: 51
            The filter width used in fringe correction, if fringe_corr is True.
        exptimekey: str, optional, default: `elaptime`
            Header keyword giving the total exposure time for all images. Used to scale off-frame to same exposure as on-frame, and normalize output to counts/s.
        target: str, optional, default: 'NGC253'
            Identifier for the science object in image. Used for calculating the heliocentric correction to wavelengths.  
            
    Returns
    -------
        hdu: astropy.io.fits.PrimaryHDU
            Reduced 2D echelle spectrum, with wavelength along x and slit along y. Saved automatically as `outimage'.   
    """
    # READ FITS IMAGE AS PRIMARYHDU
    ha=fits.open(image)[0]
    a=ha.data.copy()

    # CLEAN COSMIC RAYS
    #kern=Box2DKernel(6)
    if clean:
        print("Cleaning cosmic rays...")
        a,amask=cleancosmic(a,maxiter=3,objlim=5.0,sigfrac=0.5,sigclip=4.0)
        #a[np.where(amask==True)]=np.nan
        #a=interpolate_replace_nans(a,kern)
    # CLEAN SUBTRACTION IMAGE
    if type(offimage)==str: # then process subtraction image
        hb=fits.open(offimage)[0]# read in image
        b=hb.data.copy()
        b=ha.header[exptimekey]*(b/hb.header[exptimekey]) # normalize by exposure time, and scale to that of image
        if clean:
            print("Cleaning cosmic rays...")
            b,bmask=cleancosmic(b,maxiter=3,objlim=5.0,sigfrac=0.5,sigclip=4.0)
          #  b[np.where(bmask==True)]=np.nan
          #  b=interpolate_replace_nans(b,kern)
    else:
        b = np.zeros(a.shape)
        if type(offimage)!=type(None):
            print("Subtraction image should be of type str -- no subtraction will be performed.")
            #bmask=np.zeros(b.shape)
    
    #~~~~ DO REDUCTION / RECTIFICATION
    #~ If mode='before', do rectification BEFORE reduction (subtraction, flat-fielding)
    #~ If mode='after', do reduction first then rectification
    #~ background subtraction and fringe correction always performed after rectification/reduction
    if type(flatimage)==str:
        ff=fits.getdata(flatimage)
    else:
        ff=np.ones(a.shape)
    

    # DO REDUCTION AND RECTIFICATION       
    # if mode==0:
    print("Subtracting image pair and applying flat-field if specified...")
    flatimg,wavegrid,spatgrid=rectify(ff,spatmap,specmap)
    objimg,wavegrid,spatgrid=rectify(a,spatmap,specmap)
    skyimg,wavegrid,spatgrid=rectify(b,spatmap,specmap)
    varimg=noiseimg(objimg,flatimg,float(ha.header['elaptime']),\
           gain=gain,rdnoise=rdnoise,darkcur=darkcur)
    if mode==0:
        reducedimg,wavegrid,spatgrid=rectify((a-b)/ff,spatmap,specmap)
    elif mode==1:
        reducedimg=(objimg-skyimg)/flatimg


        
    #    reducedimg,wavegrid,spatgrid=rectify((a-b)/ff,spatmap,specmap)   
        
    # elif mode==1:
    #     print("Rectifying in spatial and spectral dimensions...")
    #     arect,wavegrid,spatgrid=rectify(a,spatmap,specmap)
    #     brect,wavegrid,spatgrid=rectify(b,spatmap,specmap)
    #     flat,wavegrid,spatgrid=rectify(ff,spatmap,specmap)
    #     reducedimg=(arect-brect)/flat   
        
        # reducedimg=reducedimg[trim_bounds[2]:trim_bounds[3],trim_bounds[0]:trim_bounds[1]]
        # wavegrid=wavegrid[trim_bounds[0]:trim_bounds[1]]
        
    # else: # only do remaining optional steps: background subtraction and/or fringe correction
    #     print("ERROR: argument `mode' must be 0 or 1 (int), returning.")
    #     return -1
    
    # trim images
    reducedimg=reducedimg[trim_bounds[2]:trim_bounds[3],trim_bounds[0]:trim_bounds[1]]
    varimg=varimg[trim_bounds[2]:trim_bounds[3],trim_bounds[0]:trim_bounds[1]]
    wavegrid=wavegrid[trim_bounds[0]:trim_bounds[1]]
    
    # PREPARE OUTPUT IMAGE HDU AND EDIT HEADER. SCALE IMAGE DATA TO COUNTS/S
    # DO HELIOCENTRIC CONVERSION AND VELOCITY. ALSO SCALE UNCERTAINTY IMAGE TO COUNTS/S
    hduout=fits.PrimaryHDU(data=reducedimg/float(ha.header[exptimekey])) 
    varimg/=float(ha.header[exptimekey])**2

    mjd=Time(ha.header['date-obs']+'T'+ha.header['utc'],format='isot',scale='utc').mjd
    tobs=Time(mjd,format='mjd')
    if target=='':
        helio=0.*u.km/u.s
    else:
        helio=heliocorr(target,tobs,observatory="Keck")
    print("Applying %.2f heliocentric correction: "%helio.value)
    spec = Spectrum1D(flux=hduout.data*u.adu/u.s,spectral_axis=wavegrid*u.um,\
                              rest_value=restwav*u.um,velocity_convention="optical")
    
    wavehelio= (wavegrid * (1. * u.dimensionless_unscaled + helio/c.to(u.km/u.s))).value
    #print(np.diff(wavehelio))
    
    print("Updating fits header...")
    ydim,xdim=reducedimg.shape
    print("SHAPE OF FINAL SPECTRUM: ",xdim,ydim)
    try:
        del hduout.header['CD1_1']
        del hduout.header['CD1_2']
        del hduout.header['CD2_2']
        del hduout.header['CD2_1']
    except:
        pass

    
    fields=['CTYPE1','CRPIX1','CRVAL1','CDELT1','CUNIT1','RESTWAV','SPECSYS','VHELIO',\
    'CTYPE2','CRPIX2','CRVAL2','CDELT2','CUNIT2','BUNIT']
    values=['WAVE',(np.arange(1,xdim)+1)[0],wavehelio[0],wavehelio[1]-wavehelio[0],'um',restwav*1.0e-6,\
    'HELIOCEN',helio.value,'XOFFSET',1.,0.,0.152,'arcsec','adu/s']

    
    for f,v in zip(fields,values):
        try:
            del hduout.header[f]
        except:
            pass
        hduout.header.set(f,v)  
        
   # hduout.header['COMMENT']='--- REDUCTION ---'   
    # hduout.header['HISTORY']="REDUCTION:"
    # hduout.header['']="pair subtraction"
    # hduout.header['']="flat-field division"
    # hduout.header['']="rectification"
    # fig=plt.figure()
    # ax=fig.add_subplot(211)
    # zlim=ZScaleInterval().get_limits(reducedimg)
    # plt.imshow(reducedimgimgimgimgimgimg,origin='lower',interpolation='None',cmap='gist_ncar',vmin=zlim[0],vmax=zlim[1])
    # ax=fig.add_subplot(212)
    # zlim=ZScaleInterval().get_limits(var)
    # plt.imshow(var,origin='lower',interpolation='None',cmap='gist_ncar',vmin=zlim[0],vmax=zlim[1])
    # plt.show()
        
    #~~~ POST PROCESSSING AND SAVING 
    # if clean: # final pixel cleaning
    #     cleaned,bpmask=cleancosmic(hduout.data,maxiter=3,objlim=5.0,sigfrac=0.5,sigclip=4.0)
    #     cleaned=interpolate_replace_nans(cleaned,Box2DKernel(4))
    #     hduout.header['']="cleaned cosmic rays"
    #     hduout=fits.PrimaryHDU(data=cleaned, header=hduout.header)
        
    # subtract residual background?
    if bkg_subtract: # subtract mean of rows that are off-source
        # bkg = ma.median(hduout.data[bkg_box_range[0]:bkg_box_range[1],:],0)
        datasub, bkg, bkgmask = fitbkg(hduout.data, order=[bkg_order,bkg_order], fitsec=bkg_sec )
      #  hduout.header['']="background subtraction"
        hduout=fits.PrimaryHDU(data=datasub, header=hduout.header)
        
        # plt.figure()
        # plt.imshow(bkg,origin='lower')
        # plt.show()
  
    if fringe_corr:
        datafringe=fringecor(hduout.data, medwidth=fringe_width, fringe_row=[0,-1], freq_bounds=fringe_freq)
     #   hduout.header['']="fringe correction"
        hduout=fits.PrimaryHDU(data=datafringe,header=hduout.header)
        

    # print("SAVING FINAL REDUCED IMAGE AS %s"%outimage)
    hduout.writeto(outimage,overwrite=True,output_verify='warn')
#     fits.writeto(outimage.split('.')[0]+"_var."+".".join(outimage.split('.')[1:]),\
#     data=varimg,header=hduout.header,overwrite=True)
    
    return hduout, varimg
    
    
#~~~~ FUNCIONALITY FOR SCAM IMAGES   
def matchscam(specfile,scamfile):
    
    # get spec start/end time in MJD
    # get start ad end times of 
    
    hspec=fits.getheader(specfile)
    hscam=[fits.getheader(f) for f in scamfile]
    dT_spec=hspec['elaptime']*u.s
   # print("Spectrum elapsed time.")
    #print hspec.keys()
    if 'UTSTART' in hspec.keys():
        Ti_spec=Time(hspec['date-obs']+'T'+hspec['utstart'],format='isot',scale='utc').mjd
        Tf_spec=Time(hspec['date-obs']+'T'+hspec['utend'],format='isot',scale='utc').mjd   
        
        # get scam time info from header
        Ti_scam=[]
        Tf_scam=[]
        for h in hscam:
            Ti_scam.append(Time(h['date-obs']+'T'+h['utstart'],format='isot',scale='utc').to_value('mjd','long'))
            Tf_scam.append(Time(h['date-obs']+'T'+h['utend'],format='isot',scale='utc').to_value('mjd','long'))
        Ti_scam=np.array(Ti_scam.to_value())
        Tf_scam=np.array(Tf_scam)
    
        match = (Ti_scam>=Ti_spec) & (Tf_scam <= Tf_spec)
      #  print("spec time start/end: ",Ti_spec,Tf_spec,Ti_scam,Ti_spec)
    else:
        Ti_spec= Time(hspec['date-obs']+'T'+hspec['utc'],scale='utc') 
        Tf_spec= Ti_spec + dT_spec
    
        #    get scam time info from header
        Ti_scam=[]
        Tf_scam=[]
        for h in hscam:
            dT=h['elaptime']*u.s
            Ti=Time(h['date-obs']+'T'+h['utc'],scale='utc')
            Tf=Ti+dT
            Ti_scam.append( Ti.to_value('mjd','long')  )
            Tf_scam.append( Tf.to_value('mjd','long') )
        Ti_scam=np.array(Ti_scam)
        Tf_scam=np.array(Tf_scam)
        #print("spec time start/end: ",Time(Ti_spec,format='mjd'),Time(Tf_spec,format='mjd'),Time(Ti_scam,format='mjd'))
        match = (Ti_scam>=Ti_spec.to_value('mjd','long')) & (Tf_scam<=Tf_spec.to_value('mjd','long')) #& (Tf_scam <= Tf_spec) # such that full scam observation must be in completed in spec time window
    
    return match.astype(np.bool)
    

def read_slitpar(image):
    """
    Returns slit center [pix,pix], slit width [''], slit len [''], slit angle on image [deg], slit position angle [deg]
    """
    h=fits.getheader(image)
    try:
        x,y=(float(h['slitcent'].split(',')[0])-1.0,float(h['slitcent'].split(',')[1])-1.0)
    except:
        x,y=(float(h['slitx']),float(h['slity']))
    slitang=float(h['slitang'])
    slitpa=float(h['slitpa'])
    slitlen=float(h['slitname'].split('x')[1])
    slitwidth=float(h['slitname'].split('x')[0])
    pixscale=float(h['spatscal'])
    return (x,y),slitwidth,slitlen,slitang,slitpa,pixscale


def mkslit(image,wgrow=2.1,lgrow=1.05):
    """
    Make a mask for the slit in input image `im`
    """
    xy,width,length,ang,pa,scale=read_slitpar(image)
    ndim=fits.getdata(image).shape
    
    slitreg = RectanglePixelRegion(PixCoord(x=xy[0],y=xy[1]),height=wgrow * (width/scale),\
    width=lgrow * (length/scale),angle=u.deg*ang)
        
    #write_ds9([slitreg],"slit_%.2fx%.2f.reg"%(width,length),coordsys='image')
    
    slitmask=slitreg.to_mask().to_image(ndim).astype(np.bool)
    
    # if saveregion:
    #     write_ds9([slitreg],"slit_%.3fx%.2f.reg"%(width,length),coordsys='image')
        
    return slitreg,slitmask
    
#~~~~ TOOLS FOR SPECTRAL AND SPATIAL EXTRACTION/FITTING

    
def fit_std_spat(data,fit_width=10,xrange=(5,1000),nodsep=12.0):
    """
    Fit Gaussian to a spatial profile to infer peak position. 
    data should be a rectified 2D spectrum
    """
    spat=np.sum(data[:,xrange[0]:xrange[1]+1],1) 
    ymax0=np.argmax(spat)
    ymin0=np.argmin(spat)
    
    # get initial estimate of central row
    dy0 = np.abs(ymax0-ymin0)
    ycent0 = np.min([ymax0,ymin0]) + float(dy0)/2.
    
    # subtract residual background
    spat = spat - np.median(spat[int(ycent0)-fit_width//2:int(ycent0)+fit_width//2 + 1])
    
    # make full y grid
    ypix=np.arange(spat.size)
    
    # first fit positive trace
    tr1 = spat[ymax0-fit_width:ymax0+fit_width+1]
    y1 = ypix[ymax0-fit_width:ymax0+fit_width+1]
    
    popt1,_=opt.curve_fit(gaussian, y1, tr1, [tr1.max(),ymax0,2.6] )
    ypos1 = popt1[1]
    
    
    fig=plt.figure()
    ax1=fig.add_subplot(121)
    ax1.plot(y1,tr1)
    ax1.plot(y1,gaussian(y1,popt1[0],popt1[1],popt1[2]))
    #plt.show()
    
    # fit negative trace
    tr2 = -1.0*spat[ymin0-fit_width:ymin0+fit_width+1]
    y2 = ypix[ymin0-fit_width:ymin0+fit_width+1]
    
    popt2,_=opt.curve_fit(gaussian, y2, tr2, [tr2.max(),ymin0,2.6] )
    ypos2 = popt2[1]
    
    ax2=fig.add_subplot(122)
    ax2.plot(y2,tr2)
    ax2.plot(y2,gaussian(y2,popt2[0],popt2[1],popt2[2]))
    plt.show()
    
    # determine central pixel and pixel scale
    dy = np.abs(ypos1 - ypos2)
    ydelta = nodsep / dy
    ycen = np.min([ypos1,ypos2]) + dy/2. 
    fwhmprof = np.mean([popt1[2],popt2[2] ])*2.3548 # convert to FWHM
    
    
    return ycen+1.0,ydelta,fwhmprof# ycen is in image coords
    
def sregister(image,slitcoord,slitpa,pixcoord,pixscale,velosys=0.,outname=None):
    """
    Given a 2D spectrum as an image hdu, the central sky coordinates and position angle of the slit,
    along with the central pixel position + scale in the 2D spectrum, 
    determine the spatial portion of the CD matrix and return the corresponding celestial WCS. This gives a way to map
    spectrum pixel coordinates (x_i, y_i) where x_i is spectrum width, of size 1 unless interpolated,
    and y_i is spectrum length, along slit length, to sky coordinates (ra_i, dec_i)
    
    Parameters
    ----------
        image : string or astropy.io.fits.PrimaryHDU
            Input spectrum image, either as a filename (string) or image HDU.
        slitcoord: astropy.coord.SkyCoord 
            Sky position of center of the slit (sky position of spectrum center). slitcoord and slitpa may be read in from a astropy sky region most easily.
        slitpa: float
            Position angle of slit/spectrum, in degrees east from north. slitpa=0. is slit along N,
            slitpa=90. is slit along E. 
        pixcoord: 2-element tuple of floats.
            Pixel coordinates of slit center, along all SPATIAL axes (not wavelength). If only 1-pixel wide along first spatial axis, and slit_length [arcsec] along second,
            then typically pixcoord = (1.0, (slit_length/pixscale)/2.0). If spectrum was interpolated along slit width such that the size along first axis is ~slit_width [arcsec]
            along first axis, then pixcoord = ( ((slit_width/pixscale)+1.0)/2.0,  (slit_length/pixscale)/2.0) ). 
        pixscale: float
            Pixel scale in units of arcseconds/pixel along slit axis in 2D spectrum. For NIRSPEC, typically 0.15-0.19[arcsec/pix]. 
            
    Returns
    -------
        astropy.io.fits.PrimaryHDU
            A 3D image HDU with WCS axes RA,DEC,WAVE/FREQ/VEL and pixel axes Z,Y,X where Z is wavelength axis, Z is slit length axis, X is slit width axis
        

    """
    
#     h=fits.open(image)[0].copy()
#     hdr=h.header
#     slit=read_ds9(slitreg)[0]
#     slitpa=slit.angle.value-90.
#     slitcoord=slit.center
#     print(slitpa,slitcoord)
    
    # create output header/wcs
#     odat=np.zeros([1,h.data.shape[0],h.data.shape[1]])
#     odat[0,:,:]=h.data
    
#     ohdu=fits.PrimaryHDU(odat)
#     ohdr=ohdu.header
    if type(image)==str:
        hdu = fits.open(image)[0]
    elif type(image)==astropy.io.fits.PrimaryHDU:
        hdu = image.copy()
    else:
        print("Invalid type for input image: must be string (filename) or astropy.io.fits.PrimaryHDU object.")
        return -1
    data=hdu.copy().data
    
    #~~~ Get spectral axis wcs and keep for later
    wcss = WCS(hdu.copy()).sub(['spectral'])
    print(wcss.wcs.naxis)
    wcss.wcs.print_contents()
    scdelt = wcss.wcs.get_cdelt()[0]
    
    #~~~ Create initial fits image hdu/header: 2-dimensional along slit spatial axes
#    print("INPUT IMAGE SHAPE: AXIS0, AXIS1 = %i, %i"%(data.shape[0],data.shape[1]))
    #pixwidth=int(pixcoord[0]*2-1)
    if len(data.shape)==3: # if 3-dimensional already, assume wavelenght is along 0th axis, Y is along 1st axis, X is along 2nd axis
        imsize = data.shape[1:]
        wavesize = data.shape[0]
    elif len(data.shape)==2: # if 2-D, assume Y along 0th axis, and Z along 1st, and construct X as size=1 axis along 2nd
        imsize = (data.shape[0], 1)
        wavesize = data.shape[1]
       # data = np.concatenate([[data]]*pixwidth,axis=0).T/float(pixwidth)
        data = np.concatenate([[data]],axis=0).T#/float(pixwidth)
    else:
        print("INVALID SHAPE OF INPUT DATA: MUST BE 2D, 3D or 1D.")
        return -1
    print("Spatial size, spectral size: (N_x,N_y)=(%i,%i), N_wave=%i"%(imsize[1],imsize[0],wavesize))
    
    
   # datempty = np.zeros(imsize) # create empty wavelength slice along spatial axes
    wcs3 = WCS(fits.PrimaryHDU(data)) # celestial WCS to be set
    #wcsc.wcs.print_contents()
    
    # first set RADESYS, EPOCH/EQUINOX, MJD-OBS, CTYPES, CUNITS
    wcs3.wcs.equinox=2000.0
    wcs3.wcs.radesys='ICRS'
#     try:
#         wcsc.wcs.mjdobs=hdr['mjd-obs']
#     except:
#         print("No header key MJD-OBS in input image. Trying with key MJD instead.")
#         wcsc.wcs.mjdobs=hdr['mjd']
        
    wcs3.wcs.ctype='RA---TAN','DEC--TAN',wcss.wcs.ctype[0]
    wcs3.wcs.cunit='deg','deg',wcss.wcs.cunit[0]
    
    # set ra axis reference pixels and reference values
    wcs3.wcs.crval=slitcoord.icrs.ra.degree,slitcoord.icrs.dec.degree,wcss.wcs.crval[0]
    wcs3.wcs.crpix=pixcoord[0],pixcoord[1],wcss.wcs.crpix[0]
    
    # using pixel scale and slit position angle, calculate CD matrix
    cdelt=pixscale/3600.
    wcs3.wcs.cd=[[-cdelt*np.cos( slitpa*np.pi/180.), cdelt*np.sin( slitpa*np.pi/180.), 0. ],
                [cdelt*np.sin( slitpa*np.pi/180.), cdelt*np.cos( slitpa*np.pi/180.), 0. ],
                [0., 0.,  scdelt ]]
#     wcsc.wcs.cdelt=pixscale/3600.,pixscale/3600.
#     wcsc.wcs.cd=[[-self.quadrant_scale.to(u.deg).value,0],
#                   [0,-self.quadrant_scale.to(u.deg).value]]
#     wcsc.wcs.crota= slitpa,slitpa #90. + slitpa
    #wcsc.wcs.get_cd()
    
    # add some additional spectral info
    wcs3.wcs.specsys=wcss.wcs.specsys
    #wcs3.wcs.ssysobs=wcss.wcs.specsys
   # wcs3.wcs.velosys=velosys*u.km/u.s
    wcs3.wcs.restwav=wcss.wcs.restwav
    wcs3.fix()

    
    # make spectral axis the 3rd axis
#     wcsreg=WCS(naxis=3)
#     wcsreg.celestial.set(wcsc)
#    # wcsreg = wcsc.sub(['celestial','spectral']) #sub(['spectral'])
#     #wcsreg = WCS( wcsc.to_header_string()+wcss.to_header_string()) 
#    # wcsreg.fix()
    wcs3.printwcs()
    
    hdr3=wcs3.to_header()
    hdr3['vhelio']=(velosys,'km/s')
    
    hdu3=fits.PrimaryHDU(data,header=hdr3)
    
    if type(outname)==str:
        hdu3.writeto(outname)

    return hdu3


def mkcube(hdus, refcoord, pixres=0.05, pixwidth=3,\
           wavebound=(4.045,4.06), wavebinfact=1,\
           imsize=(500,500), widthsmooth=1.0,\
          outname='cube.fits'):
    """ Make a spectral cube using a list of astrometrically calibrated 2D spectra (rectified).  
    
    Parameters
    ----------
        hdus : list of astropy.io.fits.PrimaryHDUs with accurate celestial,spectral WCS within the headers
            Input spectra with accurate WCS within the headers: RA,DEC,WAVE. Should be of shape (N_z,N_y,N_x) where z is along spectral axis,
            y is along the slit, x is along slit width.
        refcoord: astropy.coord.SkyCoord, default: None
            Central sky coordinates of the cube as SkyCoord. If None, will set to the mean of input central coordinates.    
        pixres: float, default: 0.1 (arcsec)
            Cube spatial pixel scale in arcseconds. 
        wavebound: 2-element tuple of floats, default: (4.04,4.07)
            The cube wavelength range in microns (low,high).  
        wavebinfact: int, default: 1
            Bin factor along wavelength direction. Will bin this many pixels together along the wavelength direction of each spectrum prior to projecting onto the cube.
            The cube wavelengths step is calculated as: wavebinfact * mean(wavedelta), where wavedelta is the wavelength steps of all spectra. 
            If 1, use average wavelength step of spectra.  
        imsize: 2-element tuple of ints
            Image size (pixels) along the spatial axes (X/RA, Y/DEC). Should be set roughly to the slit length divided by the pixel scale: L[arcsec]/pixres[arcsec/pix]
        fwhmsmooth: float, default: 1.0 arcsec
            FWHM of NaN smooth kernel in arcsec. Should be SQRT( FWHM_CUBE^2 - FWHM_INPUTS^2).

            
    Returns
    -------
        hdu: astropy.io.fits.PrimaryHDU
        
    """

    #~~~ Construct cube WCS/header
    #~~~ First get wavelength step of all spectra, and calculate wavelength grid of cube ~~~~
    wcsslist = [WCS(hdu).sub(['spectral']) for hdu in hdus]
  #  print(wcsslist[0].wcs.pc)
    
    wavedeltspec = [((ws.wcs.pc[0]*u.dimensionless_unscaled)*u.Unit(ws.world_axis_units[0])).to('um').value for ws in wcsslist]
    wavedelt = wavebinfact * np.mean(wavedeltspec) # convert to microns
    print("WAVE DELTA FOR CUBE",wavedelt)
    wavecube  = np.arange(wavebound[0], wavebound[1] + wavedelt, wavedelt ) #this is wavelength grid of output cube
    wavedim = np.size(wavecube )
    z = np.arange(wavedim)
 #   print(wavecube )
    #print(wave_cube[0])
    
    #~~~ Set reference coordinate (central coords)
#     if (refcoord==None) | (type(refcoord)!=SkyCoord):
#         refcoords = [SkyCoord(ra=h.header['crval2'],dec=h.header['crval3'],unit=(u.deg,u.deg),frame=h.header['radesys']) for h in hdu]
#         raref,decref = (np.mean([ref.icrs.ra.degree for ref in refcoords]), np.mean([ref.icrs.dec.degree for ref in refcoords]))
#     else:
    raref,decref=(refcoord.icrs.ra.degree,refcoord.icrs.dec.degree)
    print("Central coordinates of cube: ",raref,decref)   
    
    #~~~ Set up cube WCS. Define cube shape and enter WCS info in header.
    shapecube = [wavedim, imsize[1], imsize[0]]
    hducube= fits.PrimaryHDU( data=np.zeros(shapecube) )
    
    # create cube celestial and spectral axes
    hdrkeys=[('RADESYS','ICRS'),('EQUINOX',2000.0),\
             ('CTYPE1','RA---TAN'),('CUNIT1','deg'),('CRPIX1',shapecube[2]/2.),('CRVAL1',(raref,'deg')),('CDELT1',(-pixres/3600.,'deg')),\
             ('CTYPE2','DEC--TAN'),('CUNIT2','deg'),('CRPIX2',shapecube[1]/2.),('CRVAL2',(decref,'deg')),('CDELT2',(pixres/3600.,'deg')),\
             ('CTYPE3','WAVE'),('CUNIT3','um'),('CRPIX3',(z +1.0)[wavedim//2]),('CRVAL3',(wavecube[wavedim//2],'um')),('CDELT3',(np.diff(wavecube)[0],'um')),\
             ('SPECSYS','HELIOCEN'),('RESTWAV',wcsslist[0].wcs.restwav),('VHELIO',(hdus[0].header['VHELIO'],'km/s'))]#,\
           #  ('CD1_1',(-pixres/3600.,'deg')), ('CD2_2',(pixres/3600.,'deg')),('CD3_3',(np.diff(wave cube)[0].value,'um')) ]
    for hk in hdrkeys:
        hducube.header[hk[0]]=hk[1] #.set(hk[0],hk[1])
    
    wcscube = WCS(hducube)
    #wcscube.fix()
    wcscube.printwcs()
    wcsccube=wcscube.celestial

    #scube=SpectralCube.read("temp.fits").with_spectral_unit('um',velocity_convention='optical')
    cubestack=[]
    fpstack=[]
    #cubedata=np.zeros(shapecube) # to be output cube
    #cubemask=np.zeros(shapecube) # to be number of contributions to each pix
    iim=0
    for h in hdus:
        #w=WCS(h)
        #spec = np.resize(h.data,shape=[,h.data.shape[1],h.data.shape[2]])
        
        #~~ First smooth and rebin the spectrum ~~
        #~~~ 
        spdata=h.data.copy() #np.transpose(h.data)
        print("Spectrum shape: ",spdata.shape)
        # define original spectral WCS
        spwcss_orig = WCS(h).sub(['spectral'])
        spwavedim=spdata.shape[0]
        zsp=np.arange(spwavedim)
        spwave=spwcss_orig.pixel_to_world_values(zsp+1)
        spwave=spwave*u.Unit(spwcss_orig.world_axis_units[0])
        
        #~~ Calculate statistics of background away from line
        #ibg = np.where( (spwave.to(u.um).value>wavebound[1]) | (spwave.to(u.um).value<wavebound[0]))
        #spbg = np.median(spdata[:,:,ibg].ravel())
        #spbgrms = np.sqrt( np.mean( spdata[:,:,ibg]**2 ) )
        #print("Background counts",spbg)
        #spdata-=spbg
        
        #~~ smooth using 2D gaussian, by the re-bin factor in the wavelength direction and by stddev=1.0pix~0.155 arcsec in the spatial
        spsmooth=convolve(spdata[:,:,0].T,Gaussian2DKernel(x_stddev=wavebinfact/2.3548,y_stddev=1.0),normalize_kernel=True)

        #~~ Define spectrum as a Spectrum1D object for easy re-sampling/spectral smoothing
        sp=Spectrum1D(flux=spsmooth*u.dimensionless_unscaled,spectral_axis=spwave,\
                      velocity_convention='optical',rest_value=h.header['restwav']*u.m)
        
        #~~ Smooth, resample in spectral direction
        sampler = SplineInterpolatedResampler() 
#         sprebin = sampler(sp, wavecube*u.um)
        splst=[]
        for spi in sp:
#             spsm = gaussian_smooth(spi, stddev=float(wavebinfact)/2.3548)
            sprb = sampler(spi, wavecube*u.um)
            splst.append(sprb.flux.value)
        sprebindat=np.array(splst)
        
        sprebin = Spectrum1D(flux=np.array(splst)*u.dimensionless_unscaled,spectral_axis=wavecube*u.um,\
                            velocity_convention='optical',rest_value=h.header['restwav']*u.m)

        sp3data=np.transpose(np.array([sprebin.flux.value.T]*pixwidth),axes=(1,2,0))/np.float(pixwidth)
        # extend along slit width by assuming exp
        
       # print(sp3data.shape)
        #print(sprebin.flux.value.shape,sp3data.shape)
        #sp3data = np.concatenate([[sprebin.flux.value.T]]*pixwidth,axis=2)/float(pixwidth)
        #print(sp3data.shape)
        #break
    
      #  print(sprebin.flux.value)
#         plt.figure()
#         plt.plot(sprebin.wavelength.value,sprebin.flux.value.median(0))
#         plt.show()

        #~~ Now grow spectrum along slit width so that its extent is > 1 pixel (~slit width). Assume constant slit profile.
        #sp3d=np.concatenate([[sprebin.flux.value.T]]*3,axis=0).T 
        #sp3d/=float(pixwidth) # divide by pixel width, to conserve total flux (is this true?)
        #print(sp3d.shape)
        
        #~~ Get RA,DEC of all pixels in a given wavelength slice (same mapping across wavelength)
        spwcsc = WCS(h).celestial
        # adjust wcs to reflect new size of dimension along the slit width
        #spwcsc.wcs.ndim=pixwidth,
        spwcsc.wcs.crpix=float(pixwidth)/2.0 + 0.5,spwcsc.wcs.crpix[1]
        spwcsc.printwcs()
        spchdr=spwcsc.to_header()
        spchdr['naxis1']=sp3data.shape[2]
        spchdr['naxis2']=sp3data.shape[1]
        spchdr['naxis3']=sp3data.shape[0]
        sp3wcsc=WCS(spchdr).celestial
        
#         slitcoord=SkyCoord(ra=spwcsc.wcs.crval[0]*u.deg,dec=spwcsc.wcs.crval[1]*u.deg,\
#                                     frame='icrs',unit=(u.deg,u.deg))
#         slitpa=np.arctan( spwcsc.wcs.pc[0,1]/spwcsc.wcs.pc[1,1] )
#         pscale=proj_plane_pixel_scales(spwcsc)*3600.
        #print(pscale)
#         sp3wcs=WCS(fits.PrimaryHDU(sp3data))
#         sp3wcs.wcs.equinox=2000.0
#         sp3wcs.wcs.radesys='ICRS'
#         sp3wcss=wcscube.spectral(['spectral'])
#         sp3wcs.wcs.ctype='RA---TAN','DEC--TAN',sp3wcss.wcs.ctype[0]
#         sp3wcs.wcs.cunit='deg','deg',sp3wcss.wcs.cunit[0]
#         spwcs3.wcs.crval=slitcoord.icrs.ra.degree,slitcoord.icrs.dec.degree,sp3wcss.wcs.crval[0]
#         spwcs3.wcs.crpix=pixcoord[0],pixcoord[1],wcss.wcs.crpix[0]
#         cdelt=pixscale/3600.
#         wcs3.wcs.cd=[[spwcsc.wcs., cdelt*np.sin( slitpa*np.pi/180.), 0. ],
#                 [cdelt*np.sin( slitpa*np.pi/180.), cdelt*np.cos( slitpa*np.pi/180.), 0. ],
#                 [0., 0.,  scdelt ]]
#         sp3hdu = sregister(fits.PrimaryHDU(sp3data,header=h.header),\
# #                            slitcoord,slitpa,spwcsc.wcs.crpix,pscale[0],velosys=0.,outname=None)
# #         sp3wcs=WCS(sp3hdu)
# #         sp3wcs.printwcs()
        
                           
#         spreproj = reproject_interp((sp3data, spwcsc.to_header()), shape_out=shapecube,\
#                                   order='bicubic',\
#                                    return_footprint=False)
#                                   independent_celestial_slices=True)
        
#         print(spreproj.shape)
#         cubestack.append(spreproj)
        
#         xsp=np.arange(spdata.shape[2])
#         ysp=np.arange(spdata.shape[1])
#         zsp=np.arange(spdata.shape[0]) # redefine z grid to reflect rebinning
#         ixsp,iysp = np.meshgrid(xsp,ysp)
#         xxcube,yycube = astropy.wcs.utils.pixel_to_pixel(spwcsc,wcsccube,ixsp,iysp,0)
#         ixcube=np.round(xxcube).astype(np.int)
#         iycube=np.round(yycube).astype(np.int)
        
         # loop through wave slices, set the data for this image cube, and zero-valued pixels to nan
        cubedata_i = np.zeros(shapecube)
#         fp_i = np.zeros(shapecube)
        for iz in range(cubedata_i.shape[0]):
            spslice=sp3data[iz,:,:]
          #  cubedata_i[iz,:,:] = reproject_interp((spslice,spwcsc),output_projection=wcscube.celestial,\
          #                 shape_out=cubedata_i.shape[1:], order='bicubic',return_footprint=False)
          #  cubedata_i[iz,:,:] = reproject_adaptive((spslice,spwcsc),output_projection=wcscube.celestial,\
          #                 shape_out=cubedata_i.shape[1:], order='bilinear',return_footprint=False)
            cubedata_i[iz,:,:] = reproject_adaptive((spslice,spwcsc),output_projection=wcscube.celestial,\
                           shape_out=cubedata_i.shape[1:],order='bilinear',return_footprint=False)
                          
           # cubedata_i[iz,:,:] = reproject_adaptive((spslice,spwcsc),output_projection=wcscube.celestial,\
           #                shape_out=cubedata_i.shape[1:], order='bilinear',return_footprint=False)
            
            #cubedata_i[iz,:,:][iycube,ixcube] = sp3data[iz,:,:][iysp,ixsp]
        cubedata_i[np.where(cubedata_i==0.)]=np.nan
        cubestack.append(cubedata_i)
#         footprint.append
        
    cubestack=np.array(cubestack)
    cubestackmask=np.isnan(cubestack)
    cubestackma=np.ma.masked_array(cubestack,mask=cubestackmask)
    cubedata=np.nanmean(cubestack,axis=0)
    cubedata[np.where(cubedata==0.)]=np.nan
   # cubedata=np.ma.mean(cubestackma,axis=0)
    #cubedata.data[np.where(cubedata.data==0.)]=np.nan
   # cubedata=np.nanmean(cubestack,axis=0)
    #print(cubedata.shape)
    fits.writeto(outname[:-5]+"_exact.fits",cubedata,header=hducube.header,overwrite=True)
    print("REPLACING NANS")
    spatkern=Gaussian2DKernel(x_stddev=1.0,x_size=11,y_size=11)
    #spatkern=Gaussian2DKernel(x_stddev=1.0)
    #kwargs={'kind':'slinear'}
    cubeinterp=np.zeros(cubedata.shape)
    for k in range(cubedata.shape[0]):
        if (k+1)%5==0:
            print('Coadding spectral slice %i/%i'%(k+1,cubeinterp.shape[0]))
        cubeinterp[k,:,:] = interpolate_replace_nans(cubedata[k,:,:],spatkern,\
                                                     boundary='extend')    
#         cubeinterp[k,:,:] = convolve(cubedata[k,:,:],spatkern, nan_treatment='fill', boundary=None,\
#                                    fill_value=np.nan, normalize_kernel=True)
#     print(cubeinterp.shape)
    fits.writeto(outname,data=cubeinterp,header=hducube.header,overwrite=True)

        
    return fits.PrimaryHDU(cubeinterp,header=wcscube.to_header())


    
    
def centroid(spec, width, window, approx):
    p0 = max(0, approx - (window / 2))
    p1 = min(width - 1, approx + (window / 2)) + 1
    c = p0 + ndimage.center_of_mass(spec[p0:p1])[0]
    
    if abs(c - approx) > 1:
        #logger.debug('centroid error, approx = {}, centroid = {:.3f}'.format(approx, c))
        return(approx)    
    
    return(c)  
    
   

def fitgauss(x,y,par0,yerr=None,f=fitting.gaussn):
    """
        Fit 1D spectrum with simple gaussian

    """
    # print fitfunct
    # print par0,type(par0)

    paropt,parcov = opt.curve_fit(f,x,y,p0=par0,sigma=yerr,maxfev=10000)

    return paropt,parcov
    
def fitspat(s,p,par0,perr=None,f=fitting.gaussn_poly0):
    """
        Fit Spatial profile with N-Gaussians + 2nd-order polynomial

    """
    # print fitfunct
    # print par0,type(par0)

    paropt,parcov = opt.curve_fit(f,s,p,p0=par0,sigma=perr)

    return paropt,parcov
    
    
def spatextract(data,col,width,vardata=None,weights='None',combine='mean',\
				xsmooth=0.,subsky=False,skybuff=0,skywidth=50,normalize=False):
    """
    Extract 1D spatial profile (along slit) by collapsing along wavelength (x) direction.
    
    
    
    """
    ydim,xdim=data.shape
    # if smooth is non-zero, smooth along each row of the 2D spectrum
    # using a Gaussian 1D kernel with sigma=smooth [pixel]
    if xsmooth!=0.:
        kern=Gaussian1DKernel(stddev=xsmooth)
        datanew=np.zeros(data.shape)
        #print(datasmooth.shape)
        for row in range(ydim):
            datanew[row,:]=convolve(data[row,:],kern,normalize_kernel=True,nan_treatment='interpolate')
        data=datanew
    objdata=data[:,col-(width//2):col+1+(width//2)]
    goodmask= ~np.isnan(objdata)      
    skyright=data[:,col+2+(width//2)+skybuff:col+2+(width//2)+skybuff+skywidth]
    skyleft=data[:,col-(width//2)-skybuff-skywidth:col-(width//2)-skybuff]
    skydata=np.concatenate([skyright,skyleft],axis=1)
    if type(vardata)!=type(None):
        var=vardata[:,col-(width//2):col+1+(width//2)] #[row-(width//2):row+1+(width//2),:]
    else:
        #if skyside=='None':
        var=np.var(skydata)*np.ones(objdata.shape)

    # define inverse variance, signal-to-noise
    var[np.where(var<=0.)]=1.0
    ivar=1./var
    snr2=np.square(objdata)*ivar
    snr=np.sqrt(snr2)
    sigma=np.sqrt(var)
    
    # assign weights top each pixel, either no/equal weighting, S/N, (S/N)^2, or INVERSE VARIANCE
    if weights=='None':
        w = goodmask.astype(float) # equal weighting for all non-nan pixels
        err=np.sqrt( np.square(sigma).sum(1) ) / np.nansum( w, 1)
    elif weights=='snr':
        w = snr.copy()
        err=np.sqrt( np.square(w*sigma).sum(1) ) / np.nansum( w, 1)
    elif weights=='snr2':
        w=snr2.copy()
        err=np.sqrt( np.square(w*sigma).sum(1) ) / np.nansum( w, 1)
    elif weights=='var':
        w=ivar.copy()
        err=1.0/np.sqrt( np.nansum( w, 1 ) ) #np.sqrt( (sigma)**2.sum(0) ) / np.nansum( w, 0)
    else:
        print("No valid string for weights. Using inverse-variance.")
        w = goodmask.astype(float)
        err=np.sqrt( np.square(w*sigma).sum(0) ) / np.nansum( w, 0)         
    # calculate profile by collapsing along rows
#     profile= np.nansum( w * objdata,1 ) / np.nansum( w, 1)
    


    if combine=='mean':
        profile=np.nansum( w * objdata, 1) / np.nansum( w, 1)
        err=np.sqrt( np.nansum(np.square(w*sigma),1)  / np.nansum( w, 1))
    elif combine=='median':
        profile=np.nanmedian( objdata, 1)
        err=np.sqrt( np.nansum(np.square(w*sigma),1)  / np.nansum( w, 1))
    elif combine=='sum':
        profile=np.nansum( objdata, 1)
        err=np.sqrt( np.nansum(np.square(sigma),1)  )
    else:
        print("Invalid combination type, using mean.")
        profile=np.nansum(objdata,1)
        err=np.sqrt( np.nansum(np.square(sigma),1)  )
		
	    # subtract sky
    if subsky:
        profile-=np.nanmedian(skydata,1)
	    # normalize (scale error too)
    if normalize:
        err/=np.median(profile)
        profile/=np.median(profile)

#     # smooth and normalize to max
#     if smooth==0.:
#         print("Must choose smoothing factor > 0. Setting smooth=1.0.")
#         smooth=1.0
        
#     profile=convolve(profile,Gaussian1DKernel(stddev=smooth),normalize_kernel=True,nan_treatment='interpolate')
    #profile-=np.nanmin(profile)
    #proferr=convolve(proferr,Gaussian1DKernel(stddev=smooth),normalize_kernel=True,nan_treatment='interpolate')
   # profile=profile-np.nanmedian(profile)
    #proferr/=np.nansum(profile)
    #profile/=np.nansum(profile)
    
    return profile,err
        
def linextract(data,row,width,vardata=None,combine='mean',skyside='None',skybuff=0,skywidth=10,weights='None',\
    ysmooth=0.,subsky=False,display=False):
    """
    Linear extraction of spectrum using weighted averaging along spatial axis. 
    Weights can be set to the inverse variance, S/N image, or square of S/N
    """
    ydim,xdim=data.shape

    # if smooth is non-zero, smooth along each row of the 2D spectrum
    # using a Gaussian 1D kernel with sigma=smooth [pixel]
    if ysmooth!=0.:
        kern=Gaussian1DKernel(stddev=ysmooth)
        datanew=np.zeros(data.shape)
        #print(datasmooth.shape)
        for col in range(xdim):
            datanew[:,col]=convolve(data[:,col],kern,normalize_kernel=True,nan_treatment='interpolate')
        data=datanew
    # define image cut-outs for object, sky, and variance image    
    objdata=data[row-(width//2):row+1+(width//2),:]
    goodmask= ~np.isnan(objdata)
    nanmask=np.isnan(objdata) 
    if skyside=='top':
        skybounds=(row+1+(width//2)+skybuff,row+1+(width//2)+skybuff+skywidth)
        skydata=data[row+2+(width//2)+skybuff:row+2+(width//2)+skybuff+skywidth,:]
    elif skyside=='bottom':
        skybounds=(row-(width//2)-skybuff-skywidth,row-(width//2)-skybuff)
        skydata=data[row-(width//2)-skybuff-skywidth:row-(width//2)-skybuff,:]
    elif skyside=='both':
        skytop=data[row+2+(width//2)+skybuff:row+2+(width//2)+skybuff+skywidth,:]
        skybot=data[row-(width//2)-skybuff-skywidth:row-(width//2)-skybuff,:]
        skydata=np.concatenate([skytop,skybot],axis=0)
    else: #skyside=='None':
        skydata=np.zeros(objdata.shape)
        
    # calculate median sky, subtract
    if subsky:
        skymedian=np.median(skydata,0)
        sky=np.resize(skymedian,objdata.shape)
        objdata -= sky
#     else:
#         print("Enter valid string for sky")
#         return -1
#     if display:
#         plt.figure(figsize=[12,3])
#         zlim=ZScaleInterval().get_limits(data)
#         plt.imshow(data,origin='lower',vmin=zlim[0],vmax=zlim[1])
#         plt.hlines([row+width//2,row-width//2],0,data.shape[1]-1,linestyle='-')
#         plt.hlines([row,skybounds[0],skybounds[1]],0,data.shape[1]-1,linestyle='--')
#         plt.show()
        
    if type(vardata)!=type(None):
        var=vardata[row-(width//2):row+1+(width//2),:] #[row-(width//2):row+1+(width//2),:]
    else:
        if skyside=='None':
          #  print(np.nanvar(objdata))
            var=np.nanvar(objdata)*np.ones(objdata.shape)
          #  print(var)
        else:
            
            var=np.nanvar(skydata)*np.ones(objdata.shape)
    # define inverse variance, signal-to-noise
#     var[np.where(var<=0.)]=1.0
    #print(var.mean())
    ivar=1./var
    snr2=np.square(objdata)*ivar
    snr=np.sqrt(snr2)
    sigma=np.sqrt(var)
    # assign weights top each pixel, either no/equal weighting, S/N, (S/N)^2, or INVERSE VARIANCE
    if type(weights)==np.ndarray:
        w=np.zeros(objdata.shape)
        if weights.size!=objdata.shape[0]:
            print("WEIGHTS ARRAY MUST BE SAME SIZE AS WIDTH")
            return -1
        for r in range(objdata.shape[0]):
            w[r,:]=np.ones(objdata.shape[1])*weights[r]
        err=np.sqrt( np.nansum(np.square(w*sigma),0)  / np.nansum( w, 0))
    elif weights=='None':
        #w = goodmask.astype(float) # equal weighting for all non-nan pixels
        w = (~nanmask).astype(float)
        err=np.sqrt( np.nansum(np.square(w*sigma),0)  / np.nansum( w, 0))
    elif weights=='snr':
        w = snr.copy()
        err=np.sqrt( np.nansum(np.square(w*sigma),0 ) / np.nansum( w, 0))
    elif weights=='snr2':
        w=snr2.copy()
        err=np.sqrt( np.nansum(np.square(w*sigma),0 ) / np.nansum( w, 0))
    elif weights=='var':
        w=ivar.copy()
        err=1.0/np.sqrt( np.nansum( w, 0 ) ) #np.sqrt( (sigma)**2.sum(0) ) / np.nansum( w, 0)
    else:
        print("No valid string for weights. Using inverse-variance.")
        w = (~nanmask).astype(float)
        err=np.sqrt( np.nansum(np.square(w*sigma),0 ) / np.nansum( w, 0)  ) 
    #nanmask=np.isnan(objdata)           
    
    if combine=='mean':
        spec=np.nansum( w * objdata, 0) / np.nansum( w, 0)
        err=np.sqrt( np.nansum(np.square(w*sigma),0)  / np.nansum( w, 0))
    elif combine=='median':
        spec=np.nanmedian( objdata, 0)
        err=np.sqrt( np.nansum(np.square(w*sigma),0)  / np.nansum( w, 0))
    elif combine=='sum':
        spec=np.nansum( objdata, 0)
        err=np.sqrt( np.nansum(np.square(sigma),0)  )
    else:
        print("Invalid combination type, using mean.")
        spec=np.nansum(objdata,0)
        err=np.sqrt( np.nansum(np.square(sigma),0)  )
        
    # subtract sky
    #if subsky==True:
    #    spec-=np.nanmedian(skydata)

    return spec,err
    
    
    
           
# def fitprof
def optextract(data,vardata=None,row=80,width=20,spatcol=360,spatwidth=60,spatsmooth=2.5,weights='None',\
    smooth=1.0,xsmooth=1.5,thresh=25.):
    """
    Extract 1D spatial profile (along slit) by collapsing along wavelength (x) direction.
    
    
    
    """
    
    # first obtain spatial profile (smoothed, normalized to sum)
    profile,proferr=spatextract(data,vardata=vardata,col=spatcol,width=spatwidth,weights='snr',\
       smooth=spatsmooth,xsmooth=2.0,thresh=thresh)
    
    # PROFILE WILL 
    dataslice=data[row-(width//2):row+1+(width//2),:]

    ydim,xdim=dataslice.shape
    
    # if variance image supplied, slice. if not, calculate variance in background pixels. 
    dataoff=np.concatenate([data[:,col-(3*width//2):col-(width//2)], data[:,col+1+width//2:col+1+(3*width//2)]],axis=1)
    bkg=np.nanmedian(dataoff)
    #dataslice-=bkg
    if type(vardata)!=type(None):
        var=vardata[row-(width//2):row+1+(width//2),:]
    else:
        var=np.var(dataoff)*np.ones(dataslice.shape)
        
    var[np.where(var<=0.)]=1.0
    ivar=1./var
    snr2=np.square(dataslice)*ivar
    snr=np.sqrt(snr2)
    # mask pixels at too high of a signal-to-noise
    print("Peak, mean signal-to-noise in ex region, median variance: ",np.nanmax(snr),np.nanmean(snr),np.nanmedian(var))
    # set deviant pixels (>100xrms) to nan
    dataslice[np.where(snr>=thresh)]=np.nan
    
    # assign weights either to: none, S/N, (S/N)^2, or INVERSE VARIANCE
    if weights=='None':
        w=np.ones(dataslice.shape)
    elif weights=='snr':
        w=snr.copy()
    elif weights=='snr2':
        w=snr2.copy()
    elif weights=='var':
        w=ivar.copy()
    else:
        print("No valid string for weights. Using no weights.")
        w=np.ones(dataslice.shape)
    #normalize weights
    #w/=np.sum(w,1) #normalize weights
    
    # if smooth is non-zero, smooth along each row of the 2D spectrum
    # using a Gaussian 1D kernel with sigma=smooth [pixel]
    if xsmooth!=0.:
        kern=Gaussian1DKernel(stddev=xsmooth)
        datanew=np.zeros(dataslice.shape)
        #print(datasmooth.shape)
        for row in range(ydim):
            datanew[row,:]=convolve(dataslice[row,:],kern,normalize_kernel=True,nan_treatment='interpolate')
        dataslice=datanew
        
    # calculate profile by collapsing along rows
    profile= np.nansum( w* dataslice,1 ) / np.nansum( w, 1)
    proferr=np.sqrt( 1./np.nansum( np.square(w)*var, 1) ) 
    #print(proferr.mean())
        
    # smooth and normalize to max
    if smooth==0.:
        print("Must choose smoothing factor > 0. Setting smooth=1.0.")
        smooth=1.0
        
    profile=convolve(profile,Gaussian1DKernel(stddev=smooth),normalize_kernel=True,nan_treatment='interpolate')
    profile-=np.nanmin(profile)
    #proferr=convolve(proferr,Gaussian1DKernel(stddev=smooth),normalize_kernel=True,nan_treatment='interpolate')
   # profile=profile-np.nanmedian(profile)
    proferr/=np.nansum(profile)

    
    
        
    return profile,proferr   
    
# def optextract(image,waverange=(4.04,4.065),gain=5.0,rdnoise=10.,dcurr=0.67):
#
#     #~~~ Read in image, generate noise (variance) image
#     print("Not written yet")
#     return
    #~~~ get RMS of cube using off-line channels
    # offcube=cube.mask_channels( (cube.with_spectral_unit('km/s').spectral_axis < vz-(width*4.)) | \
    #        (cube.with_spectral_unit('km/s').spectral_axis > vz+(width*4.) ) )
    # #noisemap=offcube.with_mask(offcube!=0.*offcube.unit).std(axis=0)
    # rmserr=offcube.with_mask(offcube!=0.*offcube.unit).std()
    # noiserms=offcube.with_mask(offcube!=0.*offcube.unit).flattened().std()
    # #noisecube=SpectralCube(data=np.ones(cube.shape)*noiserms,\
    # #                        header=cube.hdu.header,wcs=cube.wcs,allow_huge_operations=True)
    # #offmask=subcube.get_mask_array().astype('int')
    # #npixoff=np.nansum(offmask,axis=(1,2))
    # #offcube.sum()
    # #rmsmap=offcube.std(axis=0)  # 2D
    # #errmap=offcube.mad_std(axis=0)
    # # noisemap=offcube.std(axis=0)
    # # noiserms=offcube.flattened().std()
    # #noiserms=np.nansum(np.nansum(offcube.filled_data[:],axis=1)
    # #rmscube=SpectralCube(data=np.array(noiserms*cube.shape[0] ), wcs=cube.wcs, header=cube.hdu.header)
    #
    # #~~ READ IN REGIONS CORRESPONDING OF AREA CONTAINING ALL BRACKETT SOURCES,
    # #~ AND APERTURES TO EXTRACT SPECTRUM OF EACH SOURCE
    # subreg=read_ds9(subregfil)
    # srcsubreg=read_ds9(srcregfil)
    # nsrc=len(srcsubreg)
    #
    # #~~~ Extract spectrum of full subcube, including all sources within slit bounds
    # subcube=cube.subcube_from_regions(subreg).with_spectral_unit('km/s').\
    #     spectral_slab(vz - 3*width, vz + 3*width)
    # #submask=subcube.get_mask_array().astype('int')
    # submask=subcube.get_mask_array().astype('int')
    # ngoodpix=np.nansum( submask, axis=(1,2))
    # spectot=subcube.sum(axis=(1,2))/ngoodpix.astype(np.float)
    # #espectot=subcube.mad_std(axis=(1,2)) # error on sum in N * sigma_mean
    # #espectot=subcube.mad_std((1,2))
    # # espectot=(noisecube.subcube_from_regions(subreg).with_spectral_unit('km/s').\
    # #     spectral_slab(vz - 2*width, vz + 2*width)**2).\
    # #     sum(axis=(1,2)).value**0.5
    # #espectot=noiserms*np.ones(spectot.size)
    # #espectot=noiserms*np.ones(spectot.size)#)/ngoodpix.astype(np.float)**0.5
    # # espectot/=ngoodpix.astype(np.float)
    # espectot=subcube.mad_std(axis=(1,2))
    # # nfact=np.max(spectot)*0.5
    #
    #
    # # subcont=contcube.subcube_from_regions(subreg)
    # # contsum=contcube.sum(axis=(1,2))
    #
    #
    # # save spectrum
    # xsp=pyspeckit.units.SpectroscopicAxis(subcube.spectral_axis.to(u.km/u.s).value,
    # unit='km/s', refX=4.05226, refX_unit='um', velocity_convention='optical')
    # sptot=pyspeckit.Spectrum(xarr=xsp, data=spectot, error= espectot, units='counts', header={})
    # sptot.header['RMS']=noiserms.value
    # smtot=sptot.copy()
    # smtot.smooth(smoothfact,smoothtype='boxcar',downsample=False)
    # sptot.write("spec_global_%s.fits"%outname,type='fits')
    # smtot.write("spec_global_%s_sm.fits"%outname,type='fits')
    # # np.savetxt("spec_all_"+outname+".dat",\
    # # np.vstack([subcube.with_spectral_unit(u.km/u.s).spectral_axis.value,\
    # #         spectot.value/nfact,espectot.value/nfact,ngoodpix]).T,\
    # # header="Norm factor=%.4f\n"%nfact+"RMS=%.4f in normalized cube off-chan\n"%(noiserms/nfact)+\
    # #     "Vopt(km/s)  Intensity  Intensity_Err   Npix_aper",fmt=["%.2f","%.4e","%.4e","%i"])
    # #
    #
    # # sptot= pyspeckit.Spectrum(xarr=subcube.with_spectral_unit(u.km/u.s).spectral_axis,\
    # #            data=spectot.value/nfact, error= espectot.value/nfact, \
    # #       xunits='km/s')
    # #  subplot()
    # fig1=figure(1,figsize=(5,2))
    # ax=fig1.add_subplot(111)
    # sptot.plotter(figure=fig1,axis=ax,clear=True,refresh=True,errstyle='fill')
    # sptot.plotter.axis.annotate("Full FoV",xy=(0.6,0.7),xycoords='axes fraction')
    # #sptot.plotter.axis.annotate(r"$C_{norm}=$%.4f"%nfact,xy=(0.1,0.7),xycoords='axes fraction')
    # fig1.subplots_adjust(wspace=0.0,hspace=0.0)
    # savefig("spec_all_%s.pdf"%outname,bbox_inches='tight')
    #
    # fig2=figure(2,figsize=(5,nsrc*2))
    # ax=fig2.subplots(nsrc,1)
    # for j in range(nsrc):
    #
    #     subcubej=cube.subcube_from_regions([srcsubreg[j]]).with_spectral_unit('km/s').\
    #     spectral_slab(vz - 3*width, vz + 3*width)
    #     submaskj=subcubej.get_mask_array().astype('int')
    #     # plt.figure()
    #     # plt.imshow(submaskj[10,:,:],origin='lower')
    #     # plt.show()
    #  #   ngoodpixj=np.sum( submaskj, axis=(1,2))
    #     ngoodpixj=np.nansum( submaskj, axis=(1,2))
    #     specj=subcubej.sum(axis=(1,2))/ngoodpixj.astype('float')
    #     especj=subcubej.mad_std(axis=(1,2))
    #    # especj=noiserms*np.ones(specj.size)
    #    # print(ngoodpixj,noiserms)
    #    # especj=noiserms/ngoodpixj.astype(np.float)**0.5#*np.ones(specj.size)#/ngoodpixj.astype(np.float)
    #     # especj=noiserms*np.ones(specj.size)
    #     #especj/=ngoodpixj.astype(np.float)
    #     #nfact=np.max(specj)*0.5
    #     #especj=rmscube*np.ones(specj.value.size)
    #
    #     # subcontj=contcube.subcube_from_regions([srcsubreg[j]])
    #     # contsumj=subcontj.sum(axis=(1,2))
    #
    #   #  speclist.append( (specj/nfact, especj/nfact))
    #     xspj=pyspeckit.units.SpectroscopicAxis(subcubej.spectral_axis.to(u.km/u.s).value,
    #     unit='km/s', refX=4.052262, refX_unit='um', velocity_convention='optical')
    #     spj=pyspeckit.Spectrum(xarr=xspj, data=specj, error= especj, header={})
    #     spj.header['RMS']=noiserms.value
    #     smj=spj.copy()
    #     smj.smooth(smoothfact,smoothtype='boxcar',downsample=False)
    #     spj.write("spec_%s_"%srclabels[j]+outname+".fits")
    #     smj.write("spec_%s_"%srclabels[j]+outname+"_sm.fits")
    #     # np.savetxt("spec_%s_"%srclabels[j]+outname+".dat",\
    # #     np.vstack([subcubej.with_spectral_unit(u.km/u.s).spectral_axis.value,\
    # #             specj.value/nfact,especj.value/nfact,ngoodpixj]).T,\
    # #     header="Norm factor=%.4f\n"%nfact+"RMS=%.4f in normalized cube off-chan\n"%(noiserms/nfact)+\
    # #         "Vopt(km/s)  Intensity  Intensity_Err   Npix_aper",fmt=["%.2f","%.2e","%.2e","%i"])
    # #
    # #
    # #     spj= pyspeckit.Spectrum(xarr=subcubej.with_spectral_unit(u.km/u.s).spectral_axis.value,\
    # #                    data=specj.value/nfact, error= especj.value/nfact, \
    # #               xunits='km/s')
    #     #  subplot()
    #     axj=ax[j]
    #     spj.plotter(figure=fig2,axis=axj,clear=True,refresh=True,errstyle='fill')
    #     spj.plotter.axis.annotate(srclabels[j],xy=(0.6,0.7),xycoords='axes fraction')
    #    # spj.plotter.axis.annotate(r"$C_{norm}=$%.4f"%nfact,xy=(0.1,0.7),xycoords='axes fraction')
    #     draw()
    #
    # fig2.subplots_adjust(wspace=0.0,hspace=0.0)
    # savefig("spec_aper_%s.pdf"%outname,bbox_inches='tight')
    #
  #
# def optimalExtract(*arg, **kw):
#     """
#     Extract spectrum, following Horne 1986.
#
#     :INPUTS:
#        data : 2D Numpy array
#          Appropriately calibrated frame from which to extract
#          spectrum.  Should be in units of ADU, not electrons!
#
#        variance : 2D Numpy array
#          Variances of pixel values in 'data'.
#
#        gain : scalar
#          Detector gain, in electrons per ADU
#
#        readnoise : scalar
#          Detector readnoise, in electrons.
#
#     :OPTIONS:
#        goodpixelmask : 2D numpy array
#          Equals 0 for bad pixels, 1 for good pixels
#
#        bkg_radii : 2- or 4-sequence
#          If length 2: inner and outer radii to use in computing
#          background. Note that for this to be effective, the spectral
#          trace should be positions in the center of 'data.'
#
#          If length 4: start and end indices of both apertures for
#          background fitting, of the form [b1_start, b1_end, b2_start,
#          b2_end] where b1 and b2 are the two background apertures, and
#          the elements are arranged in strictly ascending order.
#
#        extract_radius : int or 2-sequence
#          radius to use for both flux normalization and extraction.  If
#          a sequence, the first and last indices of the array to use
#          for spectral normalization and extraction.
#
#
#        dispaxis : bool
#          0 for horizontal spectrum, 1 for vertical spectrum
#
#        bord : int >= 0
#          Degree of polynomial background fit.
#
#        bsigma : int >= 0
#          Sigma-clipping threshold for computing background.
#
#        pord : int >= 0
#          Degree of polynomial fit to construct profile.
#
#        psigma : int >= 0
#          Sigma-clipping threshold for computing profile.
#
#        csigma : int >= 0
#          Sigma-clipping threshold for cleaning & cosmic-ray rejection.
#
#        finite : bool
#          If true, mask all non-finite values as bad pixels.
#
#        nreject : int > 0
#          Number of pixels to reject in each iteration.
#
#     :RETURNS:
#        3-tuple:
#           [0] -- spectrum flux (in electrons)
#
#           [1] -- uncertainty on spectrum flux
#
#           [1] -- background flux
#
#
#     :EXAMPLE:
#       ::
#
#
#     :SEE_ALSO:
#       :func:`superExtract`.
#
#     :NOTES:
#       Horne's classic optimal extraction algorithm is optimal only so
#       long as the spectral traces are very nearly aligned with
#       detector rows or columns.  It is *not* well-suited for
#       extracting substantially tilted or curved traces, for the
#       reasons described by Marsh 1989, Mukai 1990.  For extracting
#       such spectra, see :func:`superExtract`.
#     """
#
#     # 2012-08-20 08:24 IJMC: Created from previous, low-quality version.
#     # 2012-09-03 11:37 IJMC: Renamed to replace previous, low-quality
#     #                        version. Now bkg_radii and extract_radius
#     #                        can refer to either a trace-centered
#     #                        coordinate system, or the specific
#     #                        indices of all aperture edges. Added nreject.
#
#
#     from scipy import signal
#
#     # Parse inputs:
#     frame, variance, gain, readnoise = args[0:4]
#
#     # Parse options:
#     if kw.has_key('goodpixelmask'):
#         goodpixelmask = np.array(kw['goodpixelmask'], copy=True).astype(bool)
#     else:
#         goodpixelmask = np.ones(frame.shape, dtype=bool)
#
#     if kw.has_key('dispaxis'):
#         if kw['dispaxis']==1:
#             frame = frame.transpose()
#             variance = variance.transpose()
#             goodpixelmask = goodpixelmask.transpose()
#
#     if kw.has_key('verbose'):
#         verbose = kw['verbose']
#     else:
#         verbose = False
#
#     if kw.has_key('bkg_radii'):
#         bkg_radii = kw['bkg_radii']
#     else:
#         bkg_radii = [15, 20]
#         if verbose: message("Setting option 'bkg_radii' to: " + str(bkg_radii))
#
#     if kw.has_key('extract_radius'):
#         extract_radius = kw['extract_radius']
#     else:
#         extract_radius = 10
#         if verbose: message("Setting option 'extract_radius' to: " + str(extract_radius))
#
#     if kw.has_key('bord'):
#         bord = kw['bord']
#     else:
#         bord = 1
#         if verbose: message("Setting option 'bord' to: " + str(bord))
#
#     if kw.has_key('bsigma'):
#         bsigma = kw['bsigma']
#     else:
#         bsigma = 3
#         if verbose: message("Setting option 'bsigma' to: " + str(bsigma))
#
#     if kw.has_key('pord'):
#         pord = kw['pord']
#     else:
#         pord = 2
#         if verbose: message("Setting option 'pord' to: " + str(pord))
#
#     if kw.has_key('psigma'):
#         psigma = kw['psigma']
#     else:
#         psigma = 4
#         if verbose: message("Setting option 'psigma' to: " + str(psigma))
#
#     if kw.has_key('csigma'):
#         csigma = kw['csigma']
#     else:
#         csigma = 5
#         if verbose: message("Setting option 'csigma' to: " + str(csigma))
#
#     if kw.has_key('finite'):
#         finite = kw['finite']
#     else:
#         finite = True
#         if verbose: message("Setting option 'finite' to: " + str(finite))
#
#     if kw.has_key('nreject'):
#         nreject = kw['nreject']
#     else:
#         nreject = 100
#         if verbose: message("Setting option 'nreject' to: " + str(nreject))
#
#     if finite:
#         goodpixelmask *= (np.isfinite(frame) * np.isfinite(variance))
#
#
#     variance[True-goodpixelmask] = frame[goodpixelmask].max() * 1e9
#     nlam, fitwidth = frame.shape
#
#     xxx = np.arange(-fitwidth/2, fitwidth/2)
#     xxx0 = np.arange(fitwidth)
#     if len(bkg_radii)==4: # Set all borders of background aperture:
#         backgroundAperture = ((xxx0 > bkg_radii[0]) * (xxx0 <= bkg_radii[1])) + \
#             ((xxx0 > bkg_radii[2]) * (xxx0 <= bkg_radii[3]))
#     else: # Assume trace is centered, and use only radii.
#         backgroundAperture = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) <= bkg_radii[1])
#
#     if hasattr(extract_radius, '__iter__'):
#         extractionAperture = (xxx0 > extract_radius[0]) * (xxx0 <= extract_radius[1])
#     else:
#         extractionAperture = np.abs(xxx) < extract_radius
#
#     nextract = extractionAperture.sum()
#     xb = xxx[backgroundAperture]
#
#     #Step3: Sky Subtraction
#     if bord==0: # faster to take weighted mean:
#         background = an.wmean(frame[:, backgroundAperture], (goodpixelmask/variance)[:, backgroundAperture], axis=1)
#     else:
#         background = 0. * frame
#         for ii in range(nlam):
#             fit = an.polyfitr(xb, frame[ii, backgroundAperture], bord, bsigma, w=(goodpixelmask/variance)[ii, backgroundAperture], verbose=verbose-1)
#             background[ii, :] = np.polyval(fit, xxx)
#
#     # (my 3a: mask any bad values)
#     badBackground = True - np.isfinite(background)
#     background[badBackground] = 0.
#     if verbose and badBackground.any():
#         print("Found bad background values at: ", badBackground.nonzero())
#
#     skysubFrame = frame - background
#
#
#     #Step4: Extract 'standard' spectrum and its variance
#     standardSpectrum = nextract * an.wmean(skysubFrame[:, extractionAperture], goodpixelmask[:, extractionAperture], axis=1)
#     varStandardSpectrum = nextract * an.wmean(variance[:, extractionAperture], goodpixelmask[:, extractionAperture], axis=1)
#
#     # (my 4a: mask any bad values)
#     badSpectrum = True - (np.isfinite(standardSpectrum))
#     standardSpectrum[badSpectrum] = 1.
#     varStandardSpectrum[badSpectrum] = varStandardSpectrum[True - badSpectrum].max() * 1e9
#
#
#     #Step5: Construct spatial profile; enforce positivity & normalization
#     normData = skysubFrame / standardSpectrum
#     varNormData = variance / standardSpectrum**2
#
#
#     # Iteratively clip outliers
#     newBadPixels = True
#     iter = -1
#     if verbose: print("Looking for bad pixel outliers in profile construction.")
#     xl = np.linspace(-1., 1., nlam)
#
#     while newBadPixels:
#         iter += 1
#
#
#         if pord==0: # faster to take weighted mean:
#             profile = np.tile(an.wmean(normData, (goodpixelmask/varNormData), axis=0), (nlam,1))
#         else:
#             profile = 0. * frame
#             for ii in range(fitwidth):
#                 fit = an.polyfitr(xl, normData[:, ii], pord, np.inf, w=(goodpixelmask/varNormData)[:, ii], verbose=verbose-1)
#                 profile[:, ii] = np.polyval(fit, xl)
#
#         if profile.min() < 0:
#             profile[profile < 0] = 0.
#         profile /= profile.sum(1).reshape(nlam, 1)
#
#         #Step6: Revise variance estimates
#         modelData = standardSpectrum * profile + background
#         variance = (np.abs(modelData)/gain + (readnoise/gain)**2) / \
#             (goodpixelmask + 1e-9) # Avoid infinite variance
#
#         outlierSigmas = (frame - modelData)**2/variance
#         if outlierSigmas.max() > psigma**2:
#             maxRejectedValue = max(psigma**2, np.sort(outlierSigmas[:, extractionAperture].ravel())[-nreject])
#             worstOutliers = (outlierSigmas>=maxRejectedValue).nonzero()
#             goodpixelmask[worstOutliers] = False
#             newBadPixels = True
#             numberRejected = len(worstOutliers[0])
#         else:
#             newBadPixels = False
#             numberRejected = 0
#
#         if verbose: print("Rejected %i pixels on this iteration " % numberRejected)
#
#         #Step5: Construct spatial profile; enforce positivity & normalization
#         varNormData = variance / standardSpectrum**2
#
#     if verbose: print("%i bad pixels found" % iter)
#
#
#     # Iteratively clip Cosmic Rays
#     newBadPixels = True
#     iter = -1
#     if verbose: print "Looking for bad pixel outliers in optimal extraction."
#     while newBadPixels:
#         iter += 1
#
#         #Step 8: Extract optimal spectrum and its variance
#         gp = goodpixelmask * profile
#         denom = (gp * profile / variance)[:, extractionAperture].sum(1)
#         spectrum = ((gp * skysubFrame  / variance)[:, extractionAperture].sum(1) / denom).reshape(nlam, 1)
#         varSpectrum = (gp[:, extractionAperture].sum(1) / denom).reshape(nlam, 1)
#
#
#         #Step6: Revise variance estimates
#         modelData = spectrum * profile + background
#         variance = (np.abs(modelData)/gain + (readnoise/gain)**2) / \
#             (goodpixelmask + 1e-9) # Avoid infinite variance
#
#
#         #Iterate until worse outliers are all identified:
#         outlierSigmas = (frame - modelData)**2/variance
#         if outlierSigmas.max() > csigma**2:
#             maxRejectedValue = max(csigma**2, np.sort(outlierSigmas[:, extractionAperture].ravel())[-nreject])
#             worstOutliers = (outlierSigmas>=maxRejectedValues).nonzero()
#             goodpixelmask[worstOutliers] = False
#             newBadPixels = True
#             numberRejected = len(worstOutliers[0])
#         else:
#             newBadPixels = False
#             numberRejected = 0
#
#         if verbose: print "Rejected %i pixels on this iteration " % numberRejected
#
#
#     if verbose: print "%i bad pixels found" % iter
#
#     ret = (spectrum, varSpectrum, profile, background, goodpixelmask)
#
#     return  ret
#
#
#
# def superExtract(*args, **kw):
#     """
#     Optimally extract curved spectra, following Marsh 1989.
#
#     :INPUTS:
#        data : 2D Numpy array
#          Appropriately calibrated frame from which to extract
#          spectrum.  Should be in units of ADU, not electrons!
#
#        variance : 2D Numpy array
#          Variances of pixel values in 'data'.
#
#        gain : scalar
#          Detector gain, in electrons per ADU
#
#        readnoise : scalar
#          Detector readnoise, in electrons.
#
#     :OPTIONS:
#        trace : 1D numpy array
#          location of spectral trace.  If None, :func:`traceorders` is
#          invoked.
#
#        goodpixelmask : 2D numpy array
#          Equals 0 for bad pixels, 1 for good pixels
#
#        npoly : int
#          Number of profile polynomials to evaluate (Marsh's
#          "K"). Ideally you should not need to set this -- instead,
#          play with 'polyspacing' and 'extract_radius.' For symmetry,
#          this should be odd.
#
#        polyspacing : scalar
#          Spacing between profile polynomials, in pixels. (Marsh's
#          "S").  A few cursory tests suggests that the extraction
#          precision (in the high S/N case) scales as S^-2 -- but the
#          code slows down as S^2.
#
#        pord : int
#          Order of profile polynomials; 1 = linear, etc.
#
#        bkg_radii : 2-sequence
#          inner and outer radii to use in computing background
#
#        extract_radius : int
#          radius to use for both flux normalization and extraction
#
#        dispaxis : bool
#          0 for horizontal spectrum, 1 for vertical spectrum
#
#        bord : int >= 0
#          Degree of polynomial background fit.
#
#        bsigma : int >= 0
#          Sigma-clipping threshold for computing background.
#
#        tord : int >= 0
#          Degree of spectral-trace polynomial (for trace across frame
#          -- not used if 'trace' is input)
#
#        csigma : int >= 0
#          Sigma-clipping threshold for cleaning & cosmic-ray rejection.
#
#        finite : bool
#          If true, mask all non-finite values as bad pixels.
#
#        qmode : str ('fast' or 'slow')
#          How to compute Marsh's Q-matrix.  Valid inputs are
#          'fast-linear', 'slow-linear', 'fast-nearest,' 'slow-nearest,'
#          and 'brute'.  These select between various methods of
#          integrating the nearest-neighbor or linear interpolation
#          schemes as described by Marsh; the 'linear' methods are
#          preferred for accuracy.  Use 'slow' if you are running out of
#          memory when using the 'fast' array-based methods.  'Brute' is
#          both slow and inaccurate, and should not be used.
#
#        nreject : int
#          Number of outlier-pixels to reject at each iteration.
#
#        retall : bool
#          If true, also return the 2D profile, background, variance
#          map, and bad pixel mask.
#
#     :RETURNS:
#        object with fields for:
#          spectrum
#
#          varSpectrum
#
#          trace
#
#
#     :EXAMPLE:
#       ::
#
#         import spec
#         import numpy as np
#         import pylab as py
#
#         def gaussian(p, x):
#            if len(p)==3:
#                p = concatenate((p, [0]))
#            return (p[3] + p[0]/(p[1]*sqrt(2*pi)) * exp(-(x-p[2])**2 / (2*p[1]**2)))
#
#         # Model some strongly tilted spectral data:
#         nx, nlam = 80, 500
#         x0 = np.arange(nx)
#         gain, readnoise = 3.0, 30.
#         background = np.ones(nlam)*10
#         flux =  np.ones(nlam)*1e4
#         center = nx/2. + np.linspace(0,10,nlam)
#         FWHM = 3.
#         model = np.array([gaussian([flux[ii]/gain, FWHM/2.35, center[ii], background[ii]], x0) for ii in range(nlam)])
#         varmodel = np.abs(model) / gain + (readnoise/gain)**2
#         observation = np.random.normal(model, np.sqrt(varmodel))
#         fitwidth = 60
#         xr = 15
#
#         output_spec = spec.superExtract(observation, varmodel, gain, readnoise, polyspacing=0.5, pord=1, bkg_radii=[10,30], extract_radius=5, dispaxis=1)
#
#         py.figure()
#         py.plot(output_spec.spectrum.squeeze() / flux)
#         py.ylabel('(Measured flux) / (True flux)')
#         py.xlabel('Photoelectrons')
#
#
#
#     :TO_DO:
#       Iterate background fitting and reject outliers; maybe first time
#       would be unweighted for robustness.
#
#       Introduce even more array-based, rather than loop-based,
#       calculations.  For large spectra computing the C-matrix takes
#       the most time; this should be optimized somehow.
#
#     :SEE_ALSO:
#
#     """
#
#     # 2012-08-25 20:14 IJMC: Created.
#     # 2012-09-21 14:32 IJMC: Added error-trapping if no good pixels
#     #                      are in a row. Do a better job of extracting
#     #                      the initial 'standard' spectrum.
#
#     from scipy import signal
#     from pylab import *
#     from nsdata import imshow, bfixpix
#
#
#
#     # Parse inputs:
#     frame, variance, gain, readnoise = args[0:4]
#
#     frame    = gain * np.array(frame, copy=False)
#     variance = gain**2 * np.array(variance, copy=False)
#     variance[variance<=0.] = readnoise**2
#
#     # Parse options:
#     if kw.has_key('verbose'):
#         verbose = kw['verbose']
#     else:
#         verbose = False
#     if verbose: from time import time
#
#
#     if kw.has_key('goodpixelmask'):
#         goodpixelmask = kw['goodpixelmask']
#         if isinstance(goodpixelmask, str):
#             goodpixelmask = pyfits.getdata(goodpixelmask).astype(bool)
#         else:
#             goodpixelmask = np.array(goodpixelmask, copy=True).astype(bool)
#     else:
#         goodpixelmask = np.ones(frame.shape, dtype=bool)
#
#
#     if kw.has_key('dispaxis'):
#         dispaxis = kw['dispaxis']
#     else:
#         dispaxis = 0
#
#     if dispaxis==0:
#         frame = frame.transpose()
#         variance = variance.transpose()
#         goodpixelmask = goodpixelmask.transpose()
#
#
#     if kw.has_key('pord'):
#         pord = kw['pord']
#     else:
#         pord = 2
#
#     if kw.has_key('polyspacing'):
#         polyspacing = kw['polyspacing']
#     else:
#         polyspacing = 1
#
#     if kw.has_key('bkg_radii'):
#         bkg_radii = kw['bkg_radii']
#     else:
#         bkg_radii = [15, 20]
#         if verbose: message("Setting option 'bkg_radii' to: " + str(bkg_radii))
#
#     if kw.has_key('extract_radius'):
#         extract_radius = kw['extract_radius']
#     else:
#         extract_radius = 10
#         if verbose: message("Setting option 'extract_radius' to: " + str(extract_radius))
#
#     if kw.has_key('npoly'):
#         npoly = kw['npoly']
#     else:
#         npoly = 2 * int((2.0 * extract_radius) / polyspacing / 2.) + 1
#
#     if kw.has_key('bord'):
#         bord = kw['bord']
#     else:
#         bord = 1
#         if verbose: message("Setting option 'bord' to: " + str(bord))
#
#     if kw.has_key('tord'):
#         tord = kw['tord']
#     else:
#         tord = 3
#         if verbose: message("Setting option 'tord' to: " + str(tord))
#
#     if kw.has_key('bsigma'):
#         bsigma = kw['bsigma']
#     else:
#         bsigma = 3
#         if verbose: message("Setting option 'bsigma' to: " + str(bsigma))
#
#     if kw.has_key('csigma'):
#         csigma = kw['csigma']
#     else:
#         csigma = 5
#         if verbose: message("Setting option 'csigma' to: " + str(csigma))
#
#     if kw.has_key('qmode'):
#         qmode = kw['qmode']
#     else:
#         qmode = 'fast'
#         if verbose: message("Setting option 'qmode' to: " + str(qmode))
#
#     if kw.has_key('nreject'):
#         nreject = kw['nreject']
#     else:
#         nreject = 100
#         if verbose: message("Setting option 'nreject' to: " + str(nreject))
#
#     if kw.has_key('finite'):
#         finite = kw['finite']
#     else:
#         finite = True
#         if verbose: message("Setting option 'finite' to: " + str(finite))
#
#
#     if kw.has_key('retall'):
#         retall = kw['retall']
#     else:
#         retall = False
#
#
#     if finite:
#         goodpixelmask *= (np.isfinite(frame) * np.isfinite(variance))
#
#     variance[True-goodpixelmask] = frame[goodpixelmask].max() * 1e9
#     nlam, fitwidth = frame.shape
#
#     # Define trace (Marsh's "X_j" in Eq. 9)
#     if kw.has_key('trace'):
#         trace = kw['trace']
#     else:
#         trace = None
#
#     if trace is None:
#         trace = 5
#     if not hasattr(trace, '__iter__'):
#         if verbose: print "Tracing not fully tested; dispaxis may need adjustment."
#         #pdb.set_trace()
#         tracecoef = traceorders(frame, pord=trace, nord=1, verbose=verbose-1, plotalot=verbose-1, g=gain, rn=readnoise, badpixelmask=True-goodpixelmask, dispaxis=dispaxis, fitwidth=min(fitwidth, 80))
#         trace = np.polyval(tracecoef.ravel(), np.arange(nlam))
#
#
#     #xxx = np.arange(-fitwidth/2, fitwidth/2)
#     #backgroundAperture = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) < bkg_radii[1])
#     #extractionAperture = np.abs(xxx) < extract_radius
#     #nextract = extractionAperture.sum()
#     #xb = xxx[backgroundAperture]
#
#     xxx = np.arange(fitwidth) - trace.reshape(nlam,1)
#     backgroundApertures = (np.abs(xxx) > bkg_radii[0]) * (np.abs(xxx) <= bkg_radii[1])
#     extractionApertures = np.abs(xxx) <= extract_radius
#
#     nextracts = extractionApertures.sum(1)
#
#     #Step3: Sky Subtraction
#     background = 0. * frame
#     for ii in range(nlam):
#         if goodpixelmask[ii, backgroundApertures[ii]].any():
#             fit = an.polyfitr(xxx[ii,backgroundApertures[ii]], frame[ii, backgroundApertures[ii]], bord, bsigma, w=(goodpixelmask/variance)[ii, backgroundApertures[ii]], verbose=verbose-1)
#             background[ii, :] = np.polyval(fit, xxx[ii])
#         else:
#             background[ii] = 0.
#
#     background_at_trace = np.array([np.interp(0, xxx[j], background[j]) for j in range(nlam)])
#
#     # (my 3a: mask any bad values)
#     badBackground = True - np.isfinite(background)
#     background[badBackground] = 0.
#     if verbose and badBackground.any():
#         print "Found bad background values at: ", badBackground.nonzero()
#
#     skysubFrame = frame - background
#
#
#     # Interpolate and fix bad pixels for extraction of standard
#     # spectrum -- otherwise there can be 'holes' in the spectrum from
#     # ill-placed bad pixels.
#     fixSkysubFrame = bfixpix(skysubFrame, True-goodpixelmask, n=8, retdat=True)
#
#     #Step4: Extract 'standard' spectrum and its variance
#     standardSpectrum = np.zeros((nlam, 1), dtype=float)
#     varStandardSpectrum = np.zeros((nlam, 1), dtype=float)
#     for ii in range(nlam):
#         thisrow_good = extractionApertures[ii] #* goodpixelmask[ii] *
#         standardSpectrum[ii] = fixSkysubFrame[ii, thisrow_good].sum()
#         varStandardSpectrum[ii] = variance[ii, thisrow_good].sum()
#
#
#     spectrum = standardSpectrum.copy()
#     varSpectrum = varStandardSpectrum
#
#     # Define new indices (in Marsh's appendix):
#     N = pord + 1
#     mm = np.tile(np.arange(N).reshape(N,1), (npoly)).ravel()
#     nn = mm.copy()
#     ll = np.tile(np.arange(npoly), N)
#     kk = ll.copy()
#     pp = N * ll + mm
#     qq = N * kk + nn
#
#     jj = np.arange(nlam)  # row (i.e., wavelength direction)
#     ii = np.arange(fitwidth) # column (i.e., spatial direction)
#     jjnorm = np.linspace(-1, 1, nlam) # normalized X-coordinate
#     jjnorm_pow = jjnorm.reshape(1,1,nlam) ** (np.arange(2*N-1).reshape(2*N-1,1,1))
#
#     # Marsh eq. 9, defining centers of each polynomial:
#     constant = 0.  # What is it for???
#     poly_centers = trace.reshape(nlam, 1) + polyspacing * np.arange(-npoly/2+1, npoly/2+1) + constant
#
#
#     # Marsh eq. 11, defining Q_kij    (via nearest-neighbor interpolation)
#     #    Q_kij =  max(0, min(S, (S+1)/2 - abs(x_kj - i)))
#     if verbose: tic = time()
#     if qmode=='fast-nearest': # Array-based nearest-neighbor mode.
#         if verbose: tic = time()
#         Q = np.array([np.zeros((npoly, fitwidth, nlam)), np.array([polyspacing * np.ones((npoly, fitwidth, nlam)), 0.5 * (polyspacing+1) - np.abs((poly_centers - ii.reshape(fitwidth, 1, 1)).transpose(2, 0, 1))]).min(0)]).max(0)
#
#     elif qmode=='slow-linear': # Code is a mess, but it works.
#         invs = 1./polyspacing
#         poly_centers_over_s = poly_centers / polyspacing
#         xps_mat = poly_centers + polyspacing
#         xms_mat = poly_centers - polyspacing
#         Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
#         for i in range(fitwidth):
#             ip05 = i + 0.5
#             im05 = i - 0.5
#             for j in range(nlam):
#                 for k in range(npoly):
#                     xkj = poly_centers[j,k]
#                     xkjs = poly_centers_over_s[j,k]
#                     xps = xps_mat[j,k] #xkj + polyspacing
#                     xms = xms_mat[j,k] # xkj - polyspacing
#
#                     if (ip05 <= xms) or (im05 >= xps):
#                         qval = 0.
#                     elif (im05) > xkj:
#                         lim1 = im05
#                         lim2 = min(ip05, xps)
#                         qval = (lim2 - lim1) * \
#                             (1. + xkjs - 0.5*invs*(lim1+lim2))
#                     elif (ip05) < xkj:
#                         lim1 = max(im05, xms)
#                         lim2 = ip05
#                         qval = (lim2 - lim1) * \
#                             (1. - xkjs + 0.5*invs*(lim1+lim2))
#                     else:
#                         lim1 = max(im05, xms)
#                         lim2 = min(ip05, xps)
#                         qval = lim2 - lim1 + \
#                             invs * (xkj*(-xkj + lim1 + lim2) - \
#                                         0.5*(lim1*lim1 + lim2*lim2))
#                     Q[k,i,j] = max(0, qval)
#
#
#     elif qmode=='fast-linear': # Code is a mess, but it's faster than 'slow' mode
#         invs = 1./polyspacing
#         xps_mat = poly_centers + polyspacing
#         Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
#         for j in range(nlam):
#             xkj_vec = np.tile(poly_centers[j,:].reshape(npoly, 1), (1, fitwidth))
#             xps_vec = np.tile(xps_mat[j,:].reshape(npoly, 1), (1, fitwidth))
#             xms_vec = xps_vec - 2*polyspacing
#
#             ip05_vec = np.tile(np.arange(fitwidth) + 0.5, (npoly, 1))
#             im05_vec = ip05_vec - 1
#             ind00 = ((ip05_vec <= xms_vec) + (im05_vec >= xps_vec))
#             ind11 = ((im05_vec > xkj_vec) * (True - ind00))
#             ind22 = ((ip05_vec < xkj_vec) * (True - ind00))
#             ind33 = (True - (ind00 + ind11 + ind22)).nonzero()
#             ind11 = ind11.nonzero()
#             ind22 = ind22.nonzero()
#
#             n_ind11 = len(ind11[0])
#             n_ind22 = len(ind22[0])
#             n_ind33 = len(ind33[0])
#
#             if n_ind11 > 0:
#                 ind11_3d = ind11 + (np.ones(n_ind11, dtype=int)*j,)
#                 lim2_ind11 = np.array((ip05_vec[ind11], xps_vec[ind11])).min(0)
#                 Q[ind11_3d] = ((lim2_ind11 - im05_vec[ind11]) * invs * \
#                                    (polyspacing + xkj_vec[ind11] - 0.5*(im05_vec[ind11] + lim2_ind11)))
#
#             if n_ind22 > 0:
#                 ind22_3d = ind22 + (np.ones(n_ind22, dtype=int)*j,)
#                 lim1_ind22 = np.array((im05_vec[ind22], xms_vec[ind22])).max(0)
#                 Q[ind22_3d] = ((ip05_vec[ind22] - lim1_ind22) * invs * \
#                                    (polyspacing - xkj_vec[ind22] + 0.5*(ip05_vec[ind22] + lim1_ind22)))
#
#             if n_ind33 > 0:
#                 ind33_3d = ind33 + (np.ones(n_ind33, dtype=int)*j,)
#                 lim1_ind33 = np.array((im05_vec[ind33], xms_vec[ind33])).max(0)
#                 lim2_ind33 = np.array((ip05_vec[ind33], xps_vec[ind33])).min(0)
#                 Q[ind33_3d] = ((lim2_ind33 - lim1_ind33) + invs * \
#                                    (xkj_vec[ind33] * (-xkj_vec[ind33] + lim1_ind33 + lim2_ind33) - 0.5*(lim1_ind33*lim1_ind33 + lim2_ind33*lim2_ind33)))
#
#
#     elif qmode=='brute': # Neither accurate, nor memory-frugal.
#         oversamp = 4.
#         jj2 = np.arange(nlam*oversamp, dtype=float) / oversamp
#         trace2 = np.interp(jj2, jj, trace)
#         poly_centers2 = trace2.reshape(nlam*oversamp, 1) + polyspacing * np.arange(-npoly/2+1, npoly/2+1, dtype=float) + constant
#         x2 = np.arange(fitwidth*oversamp, dtype=float)/oversamp
#         Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
#         for k in range(npoly):
#             Q[k,:,:] = an.binarray((np.abs(x2.reshape(fitwidth*oversamp,1) - poly_centers2[:,k]) <= polyspacing), oversamp)
#
#         Q /= oversamp*oversamp*2
#
#     else:  # 'slow' Loop-based nearest-neighbor mode: requires less memory
#         if verbose: tic = time()
#         Q = np.zeros((npoly, fitwidth, nlam), dtype=float)
#         for k in range(npoly):
#             for i in range(fitwidth):
#                 for j in range(nlam):
#                     Q[k,i,j] = max(0, min(polyspacing, 0.5*(polyspacing+1) - np.abs(poly_centers[j,k] - i)))
#
#     if verbose: print '%1.2f s to compute Q matrix (%s mode)' % (time() - tic, qmode)
#
#
#     # Some quick math to find out which dat columns are important, and
#     #   which contain no useful spectral information:
#     Qmask = Q.sum(0).transpose() > 0
#     Qind = Qmask.transpose().nonzero()
#     Q_cols = [Qind[0].min(), Qind[0].max()]
#     nQ = len(Qind[0])
#     Qsm = Q[:,Q_cols[0]:Q_cols[1]+1,:]
#
#     # Prepar to iteratively clip outliers
#     newBadPixels = True
#     iter = -1
#     if verbose: print "Looking for bad pixel outliers."
#     while newBadPixels:
#         iter += 1
#         if verbose: print "Beginning iteration %i" % iter
#
#
#         # Compute pixel fractions (Marsh Eq. 5):
#         #     (Note that values outside the desired polynomial region
#         #     have Q=0, and so do not contribute to the fit)
#         #E = (skysubFrame / spectrum).transpose()
#         invEvariance = (spectrum**2 / variance).transpose()
#         weightedE = (skysubFrame * spectrum / variance).transpose() # E / var_E
#         invEvariance_subset = invEvariance[Q_cols[0]:Q_cols[1]+1,:]
#
#         # Define X vector (Marsh Eq. A3):
#         if verbose: tic = time()
#         X = np.zeros(N * npoly, dtype=float)
#         X0 = np.zeros(N * npoly, dtype=float)
#         for q in qq:
#             X[q] = (weightedE[Q_cols[0]:Q_cols[1]+1,:] * Qsm[kk[q],:,:] * jjnorm_pow[nn[q]]).sum()
#         if verbose: print '%1.2f s to compute X matrix' % (time() - tic)
#
#         # Define C matrix (Marsh Eq. A3)
#         if verbose: tic = time()
#         C = np.zeros((N * npoly, N*npoly), dtype=float)
#
#         buffer = 1.1 # C-matrix computation buffer (to be sure we don't miss any pixels)
#         for p in pp:
#             qp = Qsm[ll[p],:,:]
#             for q in qq:
#                 #  Check that we need to compute C:
#                 if np.abs(kk[q] - ll[p]) <= (1./polyspacing + buffer):
#                     if q>=p:
#                         # Only compute over non-zero columns:
#                         C[q, p] = (Qsm[kk[q],:,:] * qp * jjnorm_pow[nn[q]+mm[p]] * invEvariance_subset).sum()
#                     if q>p:
#                         C[p, q] = C[q, p]
#
#
#         if verbose: print '%1.2f s to compute C matrix' % (time() - tic)
#
#         ##################################################
#         ##################################################
#         # Just for reference; the following is easier to read, perhaps, than the optimized code:
#         if False: # The SLOW way to compute the X vector:
#             X2 = np.zeros(N * npoly, dtype=float)
#             for n in nn:
#                 for k in kk:
#                     q = N * k + n
#                     xtot = 0.
#                     for i in ii:
#                         for j in jj:
#                             xtot += E[i,j] * Q[k,i,j] * (jjnorm[j]**n) / Evariance[i,j]
#                     X2[q] = xtot
#
#             # Compute *every* element of C (though most equal zero!)
#             C = np.zeros((N * npoly, N*npoly), dtype=float)
#             for p in pp:
#                 for q in qq:
#                     if q>=p:
#                         C[q, p] = (Q[kk[q],:,:] * Q[ll[p],:,:] * (jjnorm.reshape(1,1,nlam)**(nn[q]+mm[p])) / Evariance).sum()
#                     if q>p:
#                         C[p, q] = C[q, p]
#         ##################################################
#         ##################################################
#
#         # Solve for the profile-polynomial coefficients (Marsh Eq. A)4:
#         if np.abs(np.linalg.det(C)) < 1e-10:
#             Bsoln = np.dot(np.linalg.pinv(C), X)
#         else:
#             Bsoln = np.linalg.solve(C, X)
#
#         Asoln = Bsoln.reshape(N, npoly).transpose()
#
#         # Define G_kj, the profile-defining polynomial profiles (Marsh Eq. 8)
#         Gsoln = np.zeros((npoly, nlam), dtype=float)
#         for n in range(npoly):
#             Gsoln[n] = np.polyval(Asoln[n,::-1], jjnorm) # reorder polynomial coef.
#
#
#         # Compute the profile (Marsh eq. 6) and normalize it:
#         if verbose: tic = time()
#         profile = np.zeros((fitwidth, nlam), dtype=float)
#         for i in range(fitwidth):
#             profile[i,:] = (Q[:,i,:] * Gsoln).sum(0)
#
#         #P = profile.copy() # for debugging
#         if profile.min() < 0:
#             profile[profile < 0] = 0.
#         profile /= profile.sum(0).reshape(1, nlam)
#         profile[True - np.isfinite(profile)] = 0.
#         if verbose: print '%1.2f s to compute profile' % (time() - tic)
#
#         #Step6: Revise variance estimates
#         modelSpectrum = spectrum * profile.transpose()
#         modelData = modelSpectrum + background
#         variance0 = np.abs(modelData) + readnoise**2
#         variance = variance0 / (goodpixelmask + 1e-9) # De-weight bad pixels, avoiding infinite variance
#
#         outlierVariances = (frame - modelData)**2/variance
#
#         if outlierVariances.max() > csigma**2:
#             newBadPixels = True
#             # Base our nreject-counting only on pixels within the spectral trace:
#             maxRejectedValue = max(csigma**2, np.sort(outlierVariances[Qmask])[-nreject])
#             worstOutliers = (outlierVariances>=maxRejectedValue).nonzero()
#             goodpixelmask[worstOutliers] = False
#             numberRejected = len(worstOutliers[0])
#             #pdb.set_trace()
#         else:
#             newBadPixels = False
#             numberRejected = 0
#
#         if verbose: print "Rejected %i pixels on this iteration " % numberRejected
#
#
#         # Optimal Spectral Extraction: (Horne, Step 8)
#         fixSkysubFrame = bfixpix(skysubFrame, True-goodpixelmask, n=8, retdat=True)
#         spectrum = np.zeros((nlam, 1), dtype=float)
#         #spectrum1 = np.zeros((nlam, 1), dtype=float)
#         varSpectrum = np.zeros((nlam, 1), dtype=float)
#         goodprof =  profile.transpose() #* goodpixelmask
#         for ii in range(nlam):
#             thisrow_good = extractionApertures[ii] #* goodpixelmask[ii]
#             denom = (goodprof[ii, thisrow_good] * profile.transpose()[ii, thisrow_good] / variance0[ii, thisrow_good]).sum()
#             if denom==0:
#                 spectrum[ii] = 0.
#                 varSpectrum[ii] = 9e9
#             else:
#                 spectrum[ii] = (goodprof[ii, thisrow_good] * skysubFrame[ii, thisrow_good] / variance0[ii, thisrow_good]).sum() / denom
#                 #spectrum1[ii] = (goodprof[ii, thisrow_good] * modelSpectrum[ii, thisrow_good] / variance0[ii, thisrow_good]).sum() / denom
#                 varSpectrum[ii] = goodprof[ii, thisrow_good].sum() / denom
#             #if spectrum.size==1218 and ii>610:
#             #    pdb.set_trace()
#
#         #if spectrum.size==1218: pdb.set_trace()
#
#     ret = baseObject()
#     ret.spectrum = spectrum
#     ret.raw = standardSpectrum
#     ret.varSpectrum = varSpectrum
#     ret.trace = trace
#     ret.units = 'electrons'
#     ret.background = background_at_trace
#
#     ret.function_name = 'spec.superExtract'
#
#     if retall:
#         ret.profile_map = profile
#         ret.extractionApertures = extractionApertures
#         ret.background_map = background
#         ret.variance_map = variance0
#         ret.goodpixelmask = goodpixelmask
#         ret.function_args = args
#         ret.function_kw = kw
#
#     return  ret
#
#
#
#
#
#
#
# #def exspec()