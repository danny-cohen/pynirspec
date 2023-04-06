#!/usr/bin/env python
# coding: utf-8

# ## <center>Basic Reduction, Grouping, Combination of NIRSPAO Slit-viewing Camera Images</center>
# ### <center>Grouping and Combining SCAM Images of NGC 253 at each unique slit position to prepare input for astrometric calibration.</center>
# ------
# 
# In this notebook I identify SCAM frames of NGC 253 + sky, perform sky-subtraction, and combine the images based on slit position, resulting in one reduced SCAM image per SPEC frame. 
# 
# Required packages:
#  ```imexam``` to infer FWHM of the PSF in the SCAM images
#  ```photutils``` for source detection
#  ```stsci``` for mosaicing, coordinate matching, astrometric registration
# 
# Source detection and image combination:
# 
# 1. Manually define science, sky SPEC image frame numbers. 
# 2. Match SCAM frames to science, sky SPEC frames using time information in SPEC image headers.
# 3. Create SCAM master sky images. 
# 4. Subtract sky, and residual background from all SCAM science frames. Trim out bad edges of all images in the process.
# 5. Combine reduced SCAM images for each SPEC frame (each slit position). 
# 
# Notes:
# - The slit PA is given by the image header DCS keyword ```ROTDEST```, held constant at $\phi=45^\circ$ 
# - Rather than matching to available reference catalogs, can run source detection on a high-resolution IR image with accurate astrometry, such as recent Hubble WFC3 imaging 
# ---

# In[6]:


import numpy as np
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from glob import glob
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.nddata import CCDData
from astropy.time import Time
from astropy.stats import mad_std
import ccdproc
from ccdproc import Combiner
from ccdproc import ImageFileCollection
from photutils import detect_sources

# Define raw image directory, list of raw SCAM files corresponding to sky (off-target) images, \
# and list of raw SCAM science files 

path='/Users/dcohen/RESEARCH/NGC253_NIRSPAO/KOA_26773/NIRSPEC/2019aug14/raw/scam/'
pre='nscam190814_'
# skyframes=['510-528','592-609'] # NIRSPEC frame numbers
# sciframes=['435-508','531-591','612-759'] # for NGC 253


# In[7]:


scamic= ImageFileCollection(path,glob_include=pre+'*.fits')

# Get all frame numbers for science and sky images 
# skyframeno=np.concatenate([[int(n) for n in np.arange(int(s.split('-')[0]),int(s.split('-')[1])+1)] for s in skyframes])
# sciframeno=np.concatenate([[int(n) for n in np.arange(int(s.split('-')[0]),int(s.split('-')[1])+1)] for s in sciframes])
# #print(skyframeno)
# skyfiles=[path+pre+"%04d.fits"%sk for sk in skyframeno]
# scifiles=[path+pre+"%04d.fits"%sc for sc in sciframeno]

# # Define frames as CCDPROC image collections
# skycol= ImageFileCollection(filenames=skyfiles,keywords=['ofname','framenum','elaptime','utstart','utend'])
# scicol= ImageFileCollection(filenames=scifiles,keywords=['ofname','framenum','elaptime','utstart','utend'])


# In[108]:


from astroscrappy import detect_cosmics
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
def cosmic_clean(data,maxiter=4,sigclip=4.0,sigfrac=1.0,objlim=5.0,mask=None):
    """
    """
#     #max_iter = 4
#     sig_clip = 4.5
#     sig_frac = 0.5
#     obj_lim = 2.0
    
    crmask,cleandata=detect_cosmics(data, inmask=mask, sigclip=sigclip, sigfrac=sigfrac, objlim=objlim, gain=2.1,                                   readnoise=10., satlevel=15000., niter=maxiter, sepmed=True,                                    cleantype='medmask', fsmode='median')
    
    #c = cosmics.cosmicsImage(data, gain=2.1, readnoise=10.0, satlevel=20000., sigclip=sig_clip, sigfrac=sig_frac, objlim=obj_lim)
    
    #c.run(maxiter=max_iter)
    return cleandata, crmask #c.cleanarray, c.mask

def match_spec_scam(hdr_spec,hdrs_scam):
    
    # get spec start/end time in MJD
    Ti_spec=Time(hdr_spec['date-obs']+'T'+hdr_spec['utstart'],format='isot',scale='utc').mjd
    Tf_spec=Time(hdr_spec['date-obs']+'T'+hdr_spec['utend'],format='isot',scale='utc').mjd   
    print(Ti_spec,Tf_spec)

    # get scam time info from header
    Ti_scam=[]
    Tf_scam=[]
    for h in hdrs_scam:
        Ti_scam.append(Time(h['date-obs']+'T'+h['utstart'],format='isot',scale='utc').mjd)
        Tf_scam.append(Time(h['date-obs']+'T'+h['utend'],format='isot',scale='utc').mjd)
    Ti_scam=np.array(Ti_scam)
    Tf_scam=np.array(Tf_scam)
    
    match = (Ti_scam>=Ti_spec) & (Tf_scam <= Tf_spec) # such that full scam observation must be in completed in spec time window
    
    return match.astype(np.bool)


# ## Match SCAM frames to SPEC frames, combine SCAM images for every slit position, i.e. every SPEC frame

# In[110]:


specpath='/Users/dcohen/RESEARCH/NGC253_NIRSPAO/KOA_26773/NIRSPEC/2019aug14/raw/spec/'
specpre='nspec190814_'
specskyframes=['57','62'] # 
#specsciframes=['43-52']# cal stars,'53-56','58-61','63-73']
specsciframes=['53-56','58-61','63-73']

scampath='/Users/dcohen/RESEARCH/NGC253_NIRSPAO/KOA_26773/NIRSPEC/2019aug14/raw/scam/'
scampre='nscam190814_'
path='/Users/dcohen/RESEARCH/NGC253_NIRSPAO/Reduction/scam/'
pathout='/Users/dcohen/RESEARCH/NGC253_NIRSPAO/Reduction/scam/reduced/'

slitmaskfile = "slit_0.41x2.26_scam_mask.fits"

# Define lists of science target, sky SPEC files
specskyframeno=[int(s) for s in specskyframes]
specsciframeno=np.concatenate([[int(n) for n in np.arange(int(s.split('-')[0]),int(s.split('-')[1])+1)]                               for s in specsciframes])
specskyfiles=[specpath+specpre+"%04d.fits"%sk for sk in specskyframeno]
specscifiles=[specpath+specpre+"%04d.fits"%sc for sc in specsciframeno]

skyic= ImageFileCollection(filenames=specskyfiles)#,keywords=['ofname','framenum','elaptime',\
                                                          #       'utstart','utend','mjd-obs','date-obs'])
objic= ImageFileCollection(filenames=specscifiles)#,keywords=['ofname','framenum','elaptime',\
                                                          #       'utstart','utend','mjd-obs','date-obs'])

scamic = ImageFileCollection(scampath,glob_include=scampre+'*.fits')

hsc=[h for h in scamic.headers()]

slitmask=fits.getdata(path+slitmaskfile)


# assuming scamic is defined to be the image collection of ALL scam images (sci+sky)
#scamskycomblist=[]
skyimages=[]
for hsp in skyic.headers():
    
    #~~ SPEC FILE 
    specfname=hsp['ofname']
    print(specfname)
    
    print(np.size(hsc))
    matches=match_spec_scam(hsp, hsc)
    print("Num matches: ",np.size(np.where(matches==True)))
    
    matched_files=scamic.summary['file'][matches]
    #print(matched_files)
    matchic=ImageFileCollection(filenames=[scampath+f for f in matched_files])
    print(matchic.summary['framenum'])
    
    #~~ COMBINE SCAM IMAGES INTO ONE IMAGE CORRESPONDING TO THIS SPEC FRAME
    images=[]
   # masks=[]
    for hdu,fname in matchic.hdus(return_fname=True):
        dat=hdu.data.copy()
        
        datclean,bpmask=cosmic_clean(dat,sigfrac=0.2,mask=slitmask)
        #bpmask=np.zeros(dat.shape)#crmask | slitmask
        
        images.append(np.ma.masked_array(datclean,mask=bpmask))
        #masks.append( bpmask )
        #mjd.append(hdu.header['mjd-obs'])
    
    imcomb=np.ma.median( images, axis=0 )
    hdrcomb=hdu.header.copy()
    hdrcomb['framenum']=hsp['framenum']
    hdrcomb['frameno']=hsp['frameno']
    hdrcomb['utstart']=hsp['utstart']
    hdrcomb['utend']=hsp['utend']
    hdrcomb['mjd-avg']=np.mean( matchic.summary['mjd'] )
    hdrcomb['frames']=",".join(["%04d"%num for num in matchic.summary['framenum']])
   # hdrcomb['elaptime']=np.mean(matchic.summary['elaptime'])
    hdrcomb['exptime']=hdrcomb['elaptime']
    outfname=pathout+'nscam_avg_s%04d.fits'%hsp['framenum']
    fits.writeto(outfname,data=imcomb.data,header=hdrcomb,overwrite=True)
    skyimages.append(imcomb)
    
skymaster=np.ma.mean(skyimages,axis=0)
hdrmaster=hdrcomb.copy()
hdrmaster['framenum']='57,62'
hdrmaster['frameno']='57,62'
fits.writeto(pathout+"nscam_avg_sky.fits",data=skymaster.data,header=hdrmaster,overwrite=True)
# fig=plt.figure(figsize=(6,6))
# ax=fig.add_subplot(111)
# ax.imshow(ccdcomb.data,origin='lower',cmap='jet',vmin=-50.,vmax=np.ma.median(ccdcomb.data)*5.0)
# plt.show()    


# ## Now loop through SPEC science (on-target) images, match the SCAM frames for that position, subtract the closest-matching sky from each frame and combine

# In[114]:


# assuming scamic is defined to be the image collection of ALL scam images (sci+sky)
skyhdus=[fits.open(pathout+'nscam_avg_s%04d.fits'%num)[0] for num in specskyframeno]
datsky=fits.getdata(pathout+'nscam_avg_sky.fits')
mjdsky=np.array([h.header['mjd-avg'] for h in skyhdus])
for hsp in objic.headers():
    
    #~~ SPEC FILE 
    specfname=hsp['ofname']
    print(specfname)
    
    #print(np.size(hsc))
    matches=match_spec_scam(hsp, hsc)
    print("Num matches: ",np.size(np.where(matches==True)))
    
    matched_files=scamic.summary['file'][matches]
    #print(matched_files)
    matchic=ImageFileCollection(filenames=[scampath+f for f in matched_files])
   # print(matchic.summary['framenum'])
    
    #~~ Reduce and combine matched SCAM frames
    images=[]
  #  masks=[]
    for hdu,fname in matchic.hdus(return_fname=True):
        dat=hdu.data.copy()
        #print(dat[:,:4].mean())
        
        # clean cosmics
        datclean,bpmask=cosmic_clean(dat,sigfrac=0.2,mask=slitmask)

        mjdobs=(Time(hdu.header['date-obs']+'T'+hdu.header['utstart'],format='isot',scale='utc').mjd+                    Time(hdu.header['date-obs']+'T'+hdu.header['utend'],format='isot',scale='utc').mjd)/2.
                
        isky=np.argmin(np.abs(mjdobs-mjdsky))
        datsky=skyhdus[isky].data
                #print("Subtracting sky: ",skyhdus[isky].header['framenum'])
            
        datred = datclean - datsky
                
        # subtract residual background calculated using top left of image
        #mean,med,rms = sigma_clipped_stats(datred, mask=bpmask , sigma=3.0, \
        #            cenfunc=np.ma.median, stdfunc=np.ma.std, maxiters=2)
        
        #datred -= med

            
      #  bpmask= crmask | slitmask
            
#         plt.figure()
#         plt.imshow(bpmask,origin='lower',cmap='binary')
#         plt.colorbar()
#         plt.show()
      
        images.append( np.ma.masked_array(datred,mask=bpmask ) )
       # masks.append( bpmask )
        #masks.append( crmask )
        #mjd.append(hdu.header['mjd-obs'])
    imcomb=np.ma.median( images, axis=0 )
    hdrcomb=hdu.header.copy()
    hdrcomb['framenum']=hsp['framenum']
    hdrcomb['frameno']=hsp['frameno']
    hdrcomb['utstart']=hsp['utstart']
    hdrcomb['utend']=hsp['utend']
    hdrcomb['mjd-avg']=np.mean( matchic.summary['mjd'] )
    hdrcomb['frames']=",".join(["%04d"%num for num in matchic.summary['framenum']])
    #hdrcomb['elaptime']=np.mean(matchic.summary['elaptime'])
    hdrcomb['exptime']=hdrcomb['elaptime']
    outfname=pathout+'nscam_avg_s%04d.fits'%hsp['framenum']
    fits.writeto(outfname,data=imcomb.data,header=hdrcomb,overwrite=True)
     


# In[ ]:




