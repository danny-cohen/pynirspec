#!/usr/bin/env python
# coding: utf-8
""" Functions for reducing and processing echelle spectra from NIRSPEC (or NIRSPAO). 

    

"""

import matplotlib, pylab, os, astropy, copy

# matplotlib.rcParams['figure.autolayout']=True
# matplotlib.rcParams['figure.figsize']=[8.0,6.0]
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings
warnings.filterwarnings(action='ignore')
import numpy.ma as ma
from scipy import ndimage as nd
from scipy import interpolate
import scipy.signal as signal
from scipy.signal import argrelextrema,find_peaks,windows
from astropy.stats import sigma_clip
import numpy.fft as fft
from astropy.io import ascii,fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.constants import c
from astropy.stats import mad_std,sigma_clipped_stats,SigmaClip
from astropy.visualization import ImageNormalize, AsinhStretch, SqrtStretch, MinMaxInterval, ZScaleInterval
from astropy.time import Time
from astropy.convolution import Gaussian2DKernel,Box2DKernel,interpolate_replace_nans,convolve
from astropy.modeling import models, fitting
from astropy.coordinates import EarthLocation
from astropy.table import Table,QTable
from astropy.coordinates import SkyCoord
from regions import read_ds9,write_ds9
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import noise_region_uncertainty,snr_threshold
from photutils import DAOStarFinder,Background2D,MedianBackground,CircularAperture
from astroscrappy import detect_cosmics
import ccdproc as ccdp
from photutils import detect_threshold, detect_sources, deblend_sources, source_properties
from photutils import CircularAperture
from photutils import DAOStarFinder,Background2D,MedianBackground
from pyraf import iraf

# ### Functions for: rectification, flat normalization, calculating the noise image, spatial and spectral rectification, converting wavelength grid to velocity in the heliocentric frame
# clean hot/cold pixels from data using astroscrappy.detect_cosmics
def cosmic_clean(data,maxiter=4,sigclip=4.5,sigfrac=0.2,objlim=5.0,mask=None,satlvl=65000.,gain=1.0,rdnoise=10.8):
    """
    cleaned_data, bad_mask = cosmic_clean(data,mask=None,...) where ... are optional arguments for detect_cosmics
    
    """
    crmask,cleandata=detect_cosmics(data, inmask=mask, sigclip=sigclip, sigfrac=sigfrac, objlim=objlim, gain=gain, \
    readnoise=rdnoise, satlevel=satlvl, niter=maxiter, sepmed=True, cleantype='medmask', fsmode='median') 
    
    return cleandata, crmask


def noise_img(objdata, flatdata, t_int, gain=2.85, rdnoise=10.8, darkcur=0.67 ):
    """
    flat is expected to be normalized and both obj and flat are expected to be rectified
    """    
    # calculate photon noise
    noisedata = objdata / gain
    
    # add read noise
    noisedata += (rdnoise / gain)**2
    
    # add dark current noise
    noisedata += (darkcur / gain) * t_int
    
    # divide by normalized flat squared
    noisedata /= flatdata**2
    
    return noisedata


def sub_poly(data,mask=None,order=[2,2]):
    """
    Fit and subtract 2D polynomial to data with x/y fit order set by `order'. 
    Used to model and subtract background, such as scattered light between echelle orders.
    
    `mask' should have value 1 for pixels to be fit, and 0 for pixels excluded from fit.
    """
    y, x = np.meshgrid(np.arange(data.shape[0]),np.arange(data.shape[1]))#[:data.shape[0], :data.shape[1]]
    p_init = models.Legendre2D(2,2)
    fit_p = fitting.LinearLSQFitter()#LevMarLSQFitter()
    p = fit_p(p_init, x[np.where(mask==0.0)], y[np.where(mask==0.0)], data[np.where(mask==0.0)])
    residual=data-p(x,y)
    return residual,p(x,y)


def sub_bkg(data,mask=None,boxsize=[1,1]):
    """
    Calculate and subtract 2D background using `photutils.Background2D'. 
    `boxsize' should have type int, or tuple/list with two ints, and is the fractional size of characteristic background 
    relative to the image size in the 0th, 1st dimensions (y, x). `boxsize=[4,3]' means the background is 1/4th, 1/3rd 
    the size of the 0th and 1st image dimensions, respectively.
    
    data_sub, background = sub_bkg(data,mask=None,boxsize=[4, 4])
    """
    bkg=Background2D(data,boxsize,mask=mask, sigma_clip=SigmaClip(sigma=2.5,maxiters=3,cenfunc=np.nanmedian,stdfunc=np.nanstd))
    
    return data-bkg.background, bkg.background

def norm_flat(data, specmap, habounds=(0,2000), lthresh=1000., badval = 1.):
    """
    data -- 2D, unrectified spectrum or image. 
    specmap -- the spectral mapping file used for rectification, from REDSPEC routine SPECMAP (UCLA)
    habounds -- column (running horizonatally) boundaries to consider for normalization. useful for clipping overscan.
    lthresh -- Pixel value marking minimum contributing to normalization  
    badval -- what value to replace masked pixels, should be 1

    returns normalized, 2D array with same size as data, with masked pixels = 1 
    """
    # make copy of data
    datacopy=data.copy()
    # make coordinate grids
    ndim=data.shape
    x, y = np.meshgrid(np.arange(ndim[0]),np.arange(ndim[1]))
    # read spectral map and get boundaries of the spectral order
    fo=open(specmap,'r')
    lim=fo.readline().split()[2:4]
    fo.close()
    vabounds=(int(lim[0]), int(lim[1]))
    # ignore pixels beyond column 1000 by setting value to 1.0
    mask = (y < vabounds[0]) | (y > vabounds[1]) | (x < habounds[0]) | (x> habounds[1]) | (datacopy < lthresh)
    # take mean of the non-masked pixels
    mean = np.ma.median(np.ma.masked_array(datacopy, mask=mask))
    # create normalized data array
    normalized = datacopy * (1-mask.astype(int))/ mean #/ mean
    # # set masked pixels to 1
    # normalized[np.where(mask==True)]=1.0
    # also mask where pixels blow up when div by mean, near edges
    normalized[np.where(normalized > 5.0)] = 1.0
    normalized[np.where(normalized < 0.2)] = 1.0
    # avoid zeroes (not to sure about these)
    normalized[np.where(normalized == 0.0)] = 1.0    
    return normalized, mean



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
        ax1=fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        # subtract smooth background and compute FFT
       # freq=np.arange(nfil,dtype='float')
        yp = np.abs( specfft[0:nfilt] )**2
        ypmax = np.max(yp)
        yp/=ypmax

        # plot power spectrum and intensity
        ax1.plot(freq,yp,'k-',lw=2)
        ax1.set_xlim(0,xdimfft)
        ax1.set_xlabel("Frequency [1/1024 pix]")
        ax1.set_ylabel("Power")
        ax1.annotate("Power Spectrum",xy=(0.75,0.75),xycoords=('axes fraction'))
        ax2.plot(xpix,specorig/specmed,'k-')
        ax2.plot(xpix,specsmooth/specmed,'b-',lw=2)
        pltymax=np.max( (specorig/specmed)[100:-100] )+0.2 
        pltymin =np.min( (specorig/specmed)[100:-100] )-0.2
        ax2.set_ylim(pltymin,pltymax)
        ax2.set_xlim(1,xdim)
        ax2.set_xlabel("Pixel")
        ax2.set_ylabel("Intensity")
        ax2.annotate("Intensity Spectrum",xy=(0.75,0.75),xycoords=('axes fraction'))
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
 
    else:
        freq_low=[x[0] for x in freq_bounds]
        freq_hi=[x[1] for x in freq_bounds]
        
    freq=np.arange(nfilt//2+1,dtype='float') / (nfilt / float(xdim) )
    nwindow=len(freq_low)
    print("Filter windows: ",len(freq_low))
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
def fringecor(data,medwidth=51,fringe_row=(10,-10)):

    """
        Remove fringe patterns in 1D spectrum using Hanning filter in FFT domain.
        Subtract a smoothed spectrum (median filter width = filtwidth), compute FFT,
        and remove the freqs in power spectrum corresponding to fringing.

    """

    #~~ name parameters that don't change throughout procedure
    xdim = data.shape[1]
    ydim = data.shape[0]

    yinit=np.median(data.copy()[fringe_row[0]:fringe_row[1],:],0)
    yinc,frq=fringe(yinit,freq_bounds=[],medwidth=medwidth,interactive=True)
   # print(yinc,frq)
        
    # loop through rows of echellogram and apply fringe correction to each on
    datacorr=np.zeros(data.shape)
    for j in range(ydim):
        datacorr[j,:],_=fringe(data.copy()[j,:],freq_bounds=frq,\
        medwidth=medwidth,interactive=False)

    return datacorr

#~~ Generate normalized, master flat-field spectrum from raw image(s).
def mkflat(flatimage,outimage,normalize=True,specmap='spec.map',normbounds=(5,2000),normthresh=1000.,clean=True,\
        darkimage=None,darkoutimage=None):
    """Form a complex number.
    
    Parameters
    ----------
        flatimage : str or list of str
            Raw flat-field image(s) to be reduced.
        outimage: str 
            Filename for output reduced flat-field image.
        normalize: bool, optional, default: True
            If False, the flat will not be mean-normalized and off-order pixels will not be replaced by value 1.
    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    # Make combined flat
    if size(flatimage)>1:
        hduout.header=fits.getheader(flatimage[0]).copy()
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
                print("Cleaning bad pixels from frame: %s"%hdrcopy['ofname'])
                datred,bpm=cosmic_clean(dat.copy(),sigfrac=0.5)
               # datcopy
            else:
                datred=dat.copy()
                bpm=np.zeros(datred.shape)
            datred[np.where(bpm==True)]=np.nan
            kern=Box2DKernel(6)
            datred=interpolate_replace_nans(datred,kern)  
            flats.append(datred)
        # combine, masking bad pixels
        flatout=np.median(flats,axis=0)
        flatitime=np.mean(itime)
    else:
        hdu=fits.open(flatimage)[0]
        dat=hdu.data
        hduout.header=hdu.header.copy()
        flatitime=hduout.header['elaptime']
        
        if clean:
            
            flatout,bpm=cosmic_clean(dat.copy(),sigfrac=0.5)
            flatout[np.where(bpm==True)]=np.nan
            kern=Box2DKernel(6)
            flatout=interpolate_replace_nans(flatout,kern)  
            
        else:
            
            flatout=dat.copy()
     
    # now do dark reduction and subtraction if a list of dark image files is supplied
    if type(darkimage)!=type(None):
        
        if size(darkimage)>1:
            hdrdark=fits.getheader(darkimage[0]).copy()
            # Make combined dark frame for flat
            darks=[] # to be list of reduced dark images
            itime=[]
            print("Cleaning and combining dark frames...")
            for f in darkimage:
        
                # read data
                hdu=fits.open(f)[0]
                dat=hdu.data
                hdr=hdu.header
                darkitime.append(hdr['elaptime'])
        
                # clean bad pixels
                if clean:
                    print("Cleaning bad pixels from frame: %s"%hdrcopy['ofname'])
                    datred,bpm=cosmic_clean(dat.copy())
                else:
                    datred=dat.copy()
                    bpm=np.zeros(dat.copy().shape)

                datred[np.where(bpm==True)]=np.nan
                kern=Box2DKernel(6)
                datred=interpolate_replace_nans(datred,kern)
        
                # append reduced data to list of reduced flats
                darks.append(datred)
               # darkbpmasks.append(bpm)

            #    darknums.append(flathdr['ofname'][-9:-5])
            # combine and save fits image
            darkout=np.mean(darks,axis=0) 
            
            darkitime=np.mean(darkitime)
            
        else:
        
            hdu=fits.open(darkimage)[0]
            dat=hdu.data
            hdrdark=hdu.header.copy()
            darkitime=hdrdark['elaptime']
        
            if clean:
            
                darkout,bpm=cosmic_clean(dat.copy(),sigfrac=0.5)
                darkout[np.where(bpm==True)]=np.nan
                kern=Box2DKernel(6)
                darkout=interpolate_replace_nans(darkout,kern)  
            
            else:
            
                darkout=dat.copy()
                
            
        flatout=flatout-darkout
        # save combined dark
        if type(darkoutimage)==str:
            hdrdark['EXPTIME']=darkitime
            fits.writeto(darkoutimage,data=darkout,header=hdrdark,overwrite=True)
    # end dark reduction
    
    # Normalize flat ?
    if normalize:
        
        flatout=norm_flat(flatout,specmap=specmap,lthresh=normthresh,habounds=normbounds)
    
    # save final reduced flat as outimage and return 
    
    fits.writeto(outimage,data=flatout,header=hduout.header,overwrite=True)

    return flatout
    

# FULL REDUCTION OF RAW NIRSPEC/NIRSPAO SPECTRA CONTAINED IN REDSPEC FUNCTION
def redspec(image, outimage, subimage=None, flatimage=None, reduce_mode=0, spatmap='spat.map',specmap='spec.map', \
            clean=True , bkg_subtract=False, bkg_row_range=(1,1), \
            fringe_corr=False, fringe_width=51, fringe_thresh=2.5,timekey='elaptime',\
            restwav=4.052262, observatory='Keck', target='NGC253' ):
    
    # READ FITS IMAGE AS PRIMARYHDU
    ha=fits.open(image)[0]
    a=ha.data.copy()

    # CLEAN COSMIC RAYS
    kern=Box2DKernel(6)
    if clean:
        print("Cleaning cosmic rays...")
        a,amask=cosmic_clean(a,maxiter=3,objlim=5.0,sigfrac=0.5,sigclip=4.0)
        a[np.where(amask==True)]=np.nan
        a=interpolate_replace_nans(a,kern)
    # CLEAN SUBTRACTION IMAGE
    if type(subimage)==str: # then process subtraction image
        hb=fits.open(subimage)[0]# read in image
        b=hb.data.copy()
        b=ha.header[timekey]*(b/hb.header[timekey]) # normalize by exposure time, and scale to that of image
        if clean:
            print("Cleaning cosmic rays...")
            b,bmask=cosmic_clean(b,maxiter=3,objlim=5.0,sigfrac=0.5,sigclip=4.0)
            b[np.where(bmask==True)]=np.nan
            b=interpolate_replace_nans(b,kern)
    else:
        if type(subimage)!=type(None):
            print("Subtraction image should be of type str -- no subtraction will be performed.")
        b = np.zeros(a.shape)
            #bmask=np.zeros(b.shape)
    
    #~~~~ DO REDUCTION / RECTIFICATION
    #~ If reduce_mode='before', do rectification BEFORE reduction (subtraction, flat-fielding)
    #~ If reduce_mode='after', do reduction first then rectification
    #~ background subtraction and fringe correction always performed after rectification/reduction
    if type(flatimage)==str:
        ff=fits.getdata(flatimage)
    else:
        ff=np.ones(a.shape)
        
    # DO REDUCTION AND RECTIFICATION       
    if reduce_mode==0:
        print("Subtracting image pair and applying flat-field if specified...")
        reduced,wavegrid,spatgrid=rectify((a-b)/ff,spatmap,specmap)   
        
        
    elif reduce_mode==1:    
        print("Rectifying in spatial and spectral dimensions...")
        arect,wavegrid,spatgrid=rectify(a,spatmap,specmap)
        brect,wavegrid,spatgrid=rectify(b,spatmap,specmap)
        ffrect,wavegrid,spatgrid=rectify(ff,spatmap,specmap)
        reduced=(arect-brect)/ffrect       
        
    else: # only do remaining optional steps: background subtraction and/or fringe correction
        print("ERROR: argument `reduce_mode' must be 0 or 1 (int), returning.")
        return -1
    
    # DISPLAY REDUCED SPECTRUM
    fig=plt.figure(figsize=(12,4))
    zlim=ZScaleInterval().get_limits(reduced)
    ax=fig.add_subplot(111)
    ax.imshow(reduced,origin='lower',interpolation='None',cmap='gist_ncar',vmin=zlim[0],vmax=zlim[1])
    plt.show(block=False)
    
     
    # PREPARE OUTPUT IMAGE HDU AND EDIT HEADER. SCALE IMAGE DATA TO COUNTS/S
    # DO HELIOCENTRIC CONVERSION AND VELOCITY 
    hduout=fits.PrimaryHDU(data=reduced/ha.header[timekey],header=ha.header.copy()) 
    print("Updating fits header...")
    ydim,xdim=reduced.shape
    try:
        del hduout.header['CD1_1']
        del hduout.header['CD1_2']
        del hduout.header['CD2_2']
        del hduout.header['CD2_1']
    except:
        print('')
        

    mjd=Time(ha.header['date-obs']+'T'+ha.header['utc'],format='isot',scale='utc').mjd
    tobs=Time(mjd,format='mjd')
    helio=heliocorr(target,tobs,observatory=observatory)
    print("Helio corr: ",helio)
    spec = Spectrum1D(flux=hduout.data*u.adu/u.s,spectral_axis=wavegrid*u.um,\
                              rest_value=restwav*u.um,velocity_convention="optical")
    
    wavehelio= (wavegrid * (1. * u.dimensionless_unscaled + helio/c.to(u.km/u.s))).value
    velobs=spec.velocity.to(u.km/u.s)
    velhelio= velobs + helio
    #print(np.diff(wavehelio),np.diff(velhelio))
    
    fields=['CTYPE1','CRPIX1','CRVAL1','CDELT1','CUNIT1','RESTWAV','SPECSYS','VHELIO',\
    'CTYPE2','CRPIX2','CRVAL2','CDELT2','CUNIT2','BUNIT']
    values=['WAVE',(np.arange(xdim)+1)[xdim//2],wavehelio[xdim//2],wavehelio[1]-wavehelio[0],'um',\
    restwav*1.0e-6,'HELIOCEN',helio.value,\
    'OFFSET',(np.arange(ydim)+1)[ydim//2], spatgrid[ydim//2],spatgrid[1]-spatgrid[0],'arcsec',\
     'adu/s']

    
    for f,v in zip(fields,values):
        try:
            del hduout.header[f]
        except:
                print('')
        hduout.header.set(f,v)  
        
    #~~~ POST PROCESSSING AND SAVING 
    if clean: # final pixel cleaning
         cleaned,bpmask=cosmic_clean(hduout.data,maxiter=2,objlim=5.0,sigfrac=0.5,sigclip=4.0)
         hduout.data=interpolate_replace_nans(cleaned,Box2DKernel(4))
        
    # subtract residual background?
    if bkg_subtract: # subtract mean of rows that are off-source
        
        bkg = np.ma.mean(hduout.data[bkg_row_range[0]:bkg_row_range[1],:],0)
        
        hduout.data -= np.resize(bkg,hduout.data.shape)

    if fringe_corr:
        
        datafringe=fringecor(hduout.data,medwidth=fringe_width)
        
        hduout.data=datafringe
        
        
        
    #~~~ Convert to heliocentric wavelength and velocity

    
    print("SAVING FINAL REDUCED IMAGE AS %s"%outimage)
    hduout.writeto(outimage,overwrite=True,output_verify='ignore')
    
    return hduout


    
def heliocorr(targ, obstime, observatory='Keck'):
    
    #obstime = Time('2017-2-14')
    if type(targ)==str: # then assume to be the target's name
        target = SkyCoord.from_name(targ)  
    elif type(targ)==astropy.coordinates.sky_coordinate.SkyCoord:
        target=targ
        
    loc = EarthLocation.of_site(observatory)  
    corr = target.radial_velocity_correction('heliocentric',obstime=obstime, location=loc).to('km/s')  
    
    return corr

def change_hobject(path,orig,new):
    
    #iraf.hedit(path,'object','.')

    iraf.hselect(path,"$I", 'object == "%s"'%orig, Stdout='tmp')
    
    iraf.hedit('@tmp',"object", new, update='yes', verify='no')
    
    os.remove('tmp')
    
    return
    
    


