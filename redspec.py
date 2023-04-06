#!/usr/bin/env python
# coding: utf-8

""" PyNIRSPEC v1.0: Reduction and Calibration of NIRSPEC (Keck II) Observations.

PRE-PROCESSING: 

 For all groups of images, such as flat-fields, object frames, sky frames, etc., must 
 be uniquely identified by the OBJECT header keyword. For example, you can set OBJECT=`FLAT` for 
 all the flat-field images to use in reduction, OBJECT=`NGC253` for all on-source frames of this galaxy, 
 OBJECT='SKY' for off-source/sky frames to subtract from on-source frames, etc. 

 spec_lib.change_hobject or pyraf.iraf.hselect/hedit can be used to change the headers.

RECTIFICATION: 

 Rectification, to produce 2D spectra with wavelength along X and the slit along Y, 
 requires spatial and spectral mapping files produced by SPATMAP.PRO and SPECMAP.PRO within the 
 IDL-based REDSPEC distribition at https://www2.keck.hawaii.edu/inst/nirspec/redspec.html.
 
 To generate the required map files enter IDL and execute PARFILE (to choose arc-lamp trace image), SPATMAP (to make 'spat.map') and SPECMAP (to make 'spec.map').

 I hope to soon translate spatmap.pro and specmap.pro to python and incorporate within the current code base.

REDUCTION STEPS: 

 0) Use SPATMAP and SPECMAP in IDL to make `spat.map` and `spec.map` files which are used for rectification.

 1) Run redspec.py: python redspec.py rawpath objname flatname --offname --darkname --clean='y' or 'n' --fringe='y' or 'n' --bgsub='y' or 'n'  

 1) Make master normalized flat-field (w optional dark combination/subtraction)

Danny Cohen, UCLA Astronmy

ver 1.0 -- April 2020
"""
import matplotlib, pylab, os, astropy
import spec_lib

#~~~~~~~~~~~~~~~~~
#~~~~~ PARAMETERS: INPUT FILES, REDUCTION OPTIONS
#~~~~~~~~~~~~~~~~~





def parse_cmnd_line_args():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Reduce raw images from NIRSPEC or NIRSPAO.')
    parser.add_argument('rawpath',help="Full path for raw input images, containing subdirectories spec/ and scam/.")
    parser.add_argument('objname',help="Identifier for science object frames-- must be identical to the value of FITS header keyword OBJECT.")
    parser.add_argument('offname',default='SKY',help="Identifier for sky/offset/subtraction frames-- must be identical to the value of FITS header keyword OBJECT.")
    parser.add_argument('flatimage',default='Flat.fits',help="Flat-field image file, produced using `spec_lib.mkflat` prior to reduction of object spectra.")
    parser.add_argument('outpath',default='./', help="Output path for reduced SPEC files.")
    parser.add_argument('spatmap',default='spat.map', help="Spatial rectification map produced by SPATMAP.")
    parser.add_argument('specmap',default='spec.map', help="Spectral rectification map produced by SPECMAP.")
    parser.add_argument('')
    parser.add_argument('', \
        help='On-target (science) SPEC frame numbers (string). Can be single frame number (ex: "50"), comma-separated list \
        of frame numbers (ex: "50,52,53"), or range of frame numbers (ex: "50-58").')
    parser.add_argument('skyframe', help='Off-target (sky) SPEC frame numbers. Same format as tarframe.')
    parser.add_argument('scamframe', help='Window of SCAM frame numbers to consider, typically "20-600". Purpose of this\
     argment is to reduce computation time.')
    parser.add_argument('outdir',  default='./', help="Output file directory.")
   # parser.add_argument('-spec',help="Specify a SPEC file to limit the SCAM files to the corresponding slit position.")
    #parser.add_argument('-reduce',help="Set to 0 to turn off reduction process.",default="yes")
    #parser.add_argument('-combine',help="Set to 0 to turn off image alignment/combination.",default="yes")
    return(parser.parse_args())
    
if __name__ == "__main__":
    """
    RUNNING SCAM REDUCTION
    """
    
    args = parse_cmnd_line_args();
    
    
    #~~ Genereate lists of all files
    tarspecfiles=frames2files(args.tarframe, filpath=args.rawdir+'spec/')
    skyspecfiles=frames2files(args.skyframe, filpath=args.rawdir+'spec/')
    scamfiles=frames2files(args.scamframe, filpath=args.rawdir+'scam/')
    
    print("List of scam image files: ",scamfiles)
   # print("List of spec target image files: ",tarspecfiles)
#    print("List of spec sky image files: ",skyspecfiles)
    
    
    #~~ Actually run reduction
    imreduce(scamfiles,tarspecfiles,skyspecfiles,args.outdir)
    





rawpath="/Users/dcohen/"


rawic=ccdp.ImageFileCollection(pathrot,glob_include=pre+'*.fits')


# In[5]:


# make masks to group images by type
# flat-fields
flats=(rawic.summary['halogen'] == 'On') & (rawic.summary['neon'] == 'Off') & (rawic.summary['argon'] == 'Off') & (rawic.summary['krypton'] == 'Off') & (rawic.summary['xenon'] == 'Off') & (rawic.summary['calmpos'] == 'In')
# dark exposures
darks = (rawic.summary['halogen'] == 'Off') & (rawic.summary['neon'] == 'Off') & (rawic.summary['argon'] == 'Off') & (rawic.summary['krypton'] == 'Off') & (rawic.summary['xenon'] == 'Off') & (rawic.summary['calmpos'] == 'In')
# arc-lamp spectra
arcs = (rawic.summary['halogen'] == 'Off') & ( (rawic.summary['neon'] == 'On') | (rawic.summary['argon'] == 'On') | (rawic.summary['krypton'] == 'On') | (rawic.summary['xenon'] == 'On') )& (rawic.summary['calmpos'] == 'In')
# calibration/standard stars (HD3604 and HD225200)
stdstar1 = (rawic.summary['object'] == ' HD225200')
stdstar2 = (rawic.summary['object'] == ' HD3604')
# science object (NGC 253) and sky frames
obj= (rawic.summary['object'] == ' N253') | (rawic.summary['object'] == ' N253 TH2')
sky= (rawic.summary['object'] == ' Sky')
#

# Define file lists
path=pathrot
darkfiles=[path+f for f in rawic.summary['file'][darks]]
flatfiles=[path+f for f in rawic.summary['file'][flats]]
arcfiles=[path+f for f in rawic.summary['file'][arcs]]
std1files=[path+f for f in rawic.summary['file'][stdstar1]]
std2files=[path+f for f in rawic.summary['file'][stdstar2]]
objfiles=[path+f for f in rawic.summary['file'][obj]]
skyfiles=[path+f for f in rawic.summary['file'][sky]]

# Define image file collections (ccdproc)
flatic = ccdp.ImageFileCollection(filenames=flatfiles)
darkic = ccdp.ImageFileCollection(filenames=darkfiles)
arcic= ccdp.ImageFileCollection(filenames=arcfiles)
std1ic=ccdp.ImageFileCollection(filenames=std1files)
std2ic=ccdp.ImageFileCollection(filenames=std2files)
objic=ccdp.ImageFileCollection(filenames=objfiles)
skyic=ccdp.ImageFileCollection(filenames=skyfiles)


# In[6]:


order=read_ds9("order19.reg")
ndim=fits.getdata(darkic.files[0]).shape #.data.shape
ordermask=order[0].to_mask().to_image(ndim)
rinter=read_ds9("spec/cal/interorder.reg")#.to_mask().to_image(ndim)
interorder=np.zeros(ndim)
for r in rinter:
    #r.to_mask().to_image(ndim)
    interorder+=r.to_mask().to_image(ndim)
interorder[np.where(interorder>0)]/=interorder[np.where(interorder>0)]
#offorder=~ordermask.astype(np.bool)
plt.imshow(interorder,cmap='binary',origin='lower')
plt.show()


# ### Make master flat, trace orders, fit and subtract scattered light using inter-order regions
# #### Start by making the combined dark frame for the flat
# 
# #### Note: flattening usually occurs before rectification of spectra, but for NIRSPAO I have found that doing rectification prior to flattening can result in cleaner spectra. Try it both ways.

# In[ ]:


# get exp time of flats and make list of dark files
exptime=fits.getheader(flatic.files[-1])['elaptime']
darks=darkic.files_filtered(**{'elaptime':exptime})[1:7]
flats=flatic.files[-8:]

#print(darks,flats)

# run reduce_flat to reduce darks and make dark-subtracted, combined flat
reduce_flat(flats,darks,outflat="spec/cal/Flat.fits",outdark="spec/cal/Dark.fits",                      clean=True,dark_subtract=True,flat_normalize=True,ordermask=ordermask)
#hrot=h.copy()
# frot=np.rot90(flat,-1)
# drot=np.rot90(dark,-1)
# fits.writeto("spec/cal/Flatn_rot.fits",data=frot,overwrite=True)
# fits.writeto("spec/cal/Dark_rot.fits",data=drot,overwrite=True)


# In[ ]:


zlim=ZScaleInterval().get_limits(flat)
plt.imshow(flat,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])


# In[ ]:


flat=fits.getdata("spec/cal/Flat.fits")
dark=fits.getdata("spec/cal/Dark.fits")
#interorder=read_ds9("spec/cal/interorder.reg")[0].to_mask().to_image(flat.shape)
#flatinter=flat*interorders.astype(int)
datasub,scattermod=sub_scatter(flat,mask=1.0-interorder.astype(int))
fits.writeto("spec/cal/Flat.sctsub.fits",data=datasub,header=fits.getheader("spec/cal/Flat.fits"),overwrite=True)
flatnorm,flatscale=normalize(datasub,ordermask)
fits.writeto("spec/cal/NFlat.sctsub.fits",data=flatnorm,header=fits.getheader("spec/cal/NFlat.fits"),overwrite=True)
#zlim=ZScaleInterval().get_limits()
fig=plt.figure()
ax1=fig.add_subplot(131)
zlim=ZScaleInterval().get_limits(flat*interorder.astype(int))
ax1.imshow(flat*interorder.astype(int),origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])
ax2=fig.add_subplot(132)
zlim=ZScaleInterval().get_limits(scattermod*interorder)
ax2.imshow(scattermod*interorder,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])
ax3=fig.add_subplot(133)
zlim=ZScaleInterval().get_limits(datasub)
ax3.imshow(datasub,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])
plt.show()
#plt.imshow(scattermod,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])
#nflatsub,nmask,nscale=normalize(flat,ordermask)
#fits.writeto("spec/cal/nFlat.fits",data=nflatsub,overwrite=True)
# flatrect,wavearr=rectify(flat,"spec/spat.map","spec/spec.map")
# #flatmask=np.zeros(flatrect.shape)
# #flatrect/=np.median(flatrect[5:-5,10:1990])
# mean=np.median(flatrect[6:270,10:2001])
# print("norm factor: ",mean)
# flatnorm=flatrect/mean
# flatnorm[np.where(flatnorm>5.0)]=1
# flatnorm[np.where(flatnorm < 0.2)]=1
# #flatnorm/=np.median(flatnorm[np.where(flatnorm!=1.0)])
# #flatnorm[np.where(flatnorm==0.0)]=1


# In[ ]:



flat=fits.getdata("spec/cal/NFlat.fits")
flatrect,wavegrid=rectify(flat,'spec/spat.map','spec/spec.map',mask=(flat==1.0))
#flatrect==fits.getdata("spec/cal/NFlat.rect.fits")
#flat=fits.getdata("spec/cal/nFlat_rect.fits")

hred=reduce_obj(arcic.files[-1],darkic.files[-1],"spec/cal/NFlat.fits",clean=True,                   flatten=True,rectify_image=True,spatmap='spec/spat.map',specmap='spec/spec.map',                   bkg_subtract=False)
hred.writeto("spec/cal/Arc.1.fits",overwrite=True)


hrect=reduce_obj(arcic.files[-1],darkic.files[-1],"spec/cal/NFlat.fits",clean=True,                   flatten=False,rectify_image=True,spatmap='spec/spat.map',specmap='spec/spec.map',                   bkg_subtract=False)
hred2=fits.PrimaryHDU(data=hrect.data/flatrect,header=hrect.header)
hred2.writeto("spec/cal/Arc.2.fits",overwrite=True)

fig=plt.figure()
ax1=fig.add_subplot(211)
zlim=ZScaleInterval().get_limits(hred.data)
ax1.imshow(hred.data,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])
ax2=fig.add_subplot(212)
zlim=ZScaleInterval().get_limits(hred2.data)
ax2.imshow(hred2.data,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])
# ax3=fig.add_subplot(413)
# zlim=ZScaleInterval().get_limits(bg2)
# ax3.imshow(bg2,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])
# ax4=fig.add_subplot(414)
# zlim=ZScaleInterval().get_limits(datred2)
# ax4.imshow(datred2,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])
# plt.show()

# plt.figure()
# plt.plot(np.mean(datred[130:140,6:2010],axis=0),'r-')
# #plt.plot(np.sum(harcred.data[130:140,:],axis=0),'r--')
# plt.plot(np.mean(datred2[130:140,6:2010],axis=0),'b-')
# plt.ylim(-5.0,10.0)
#plt.plot(np.sum(datflat[130:140,:],axis=0),'b--')
# zlim=ZScaleInterval().get_limits(harcred.data)
plt.show()


# In[ ]:





# In[ ]:


# Reduce cal star
# read hdus, select ABBA pattern,
print(std1ic.summary['file'],std1ic.summary['elaptime'])
print(std2ic.summary['file'],std2ic.summary['elaptime'])

flat=fits.getdata("spec/cal/NFlat.fits")
#flatr=fits.getdata('spec/cal/NFlat.rect.fits')
#hdark=fits.open(darkic.summary['file'][-1])[0]

# Reduce star 1
hdus=[h for h in std1ic.hdus()][1:] # first exposure is bad
print(hdus[0].header['ofname'])
#hdark.data*=(hdus[0].header['elaptime']/hdark.header['elaptime'])
hduAB=reduce_obj(std1ic.files[2],std1ic.files[1],"spec/cal/NFlat.fits",clean=True,flatten=True,                  rectify_image=True,bkg_subtract=False,                  spatmap='spec/spat.map',specmap='spec/spec.map')

#hduA=reduce_obj(hdus[0],hdark,flat,clean=False,flatten=False,rotate_image=True,rectify_image=False,bkg_subtract=False,statbox=statbox)
#hduB=reduce_obj(hdus[1],hdark,flat,clean=False,flatten=False,rotate_image=True,rectify_image=False,bkg_subtract=False,statbox=statbox)
#hduAB2=reduce_obj(hdus[2],hdus[3],flat,clean=False,flatten=True,rotate_image=True,rectify_image=False,bkg_subtract=False,statbox=statbox)
#hduAB=reduce_obj(hdus[0],hdus[1],flat,clean=False,flatten=True,rectify_image=True)
#fits.writeto("spec/cal/hd225200_ff_rot.A.fits",data=hduA.data,header=hduA.header,overwrite=True)
#fits.writeto("spec/cal/hd225200_ff_rot.An.fits",data=-1.0*hduA.data,header=hduA.header,overwrite=True)
#fits.writeto("spec/cal/hd225200_ff_rot.B.fits",data=hduB.data,header=hduB.header,overwrite=True)
#fits.writeto("spec/cal/hd225200_ff_rot.Bn.fits",data=-1.0*hduB.data,header=hduB.header,overwrite=True)
fits.writeto("spec/hd225200_A-B.fits",data=hduAB.data,header=hduAB.header,overwrite=True)
#fits.writeto("spec/cal/hd225200_ff_rot.2.fits",data=hduAB2.data,header=hduAB1.header)
#datAB=np.mean([hduAB1.data,hduAB2.data],axis=0)
#fits.writeto("spec/cal/hd225200_ff_rot.fits",data=datAB,header=hduAB1.header,overwrite=True)
#fits.writeto("spec/cal/hd225200_ff_rot_neg.fits",data=-1.0*datAB,header=hduAB1.header,overwrite=True)


# ## Form combined sky spectrum, and reduce object spectra

# In[ ]:


from astropy.convolution import interpolate_replace_nans
help(interpolate_replace_nans)

kern=Box2DKernel(5)
hsky=[h for h in skyic.hdus()]
skydat=[]
skymask=[]
mjdsky=[]
for h in hsky:
    print("Image: ",h.header['ofname'])
    dat=h.data.copy()
    print("Exposure time: ",h.header['elaptime'])
    #print("Cleaning CRs...")
    #print(h.header['date-obs']+'T'+h.header['utstart'])
    mjdsky.append(Time(h.header['date-obs']+'T'+h.header['utstart'],format='isot').mjd)
   # print(tobs.mjd)
  #  datcl,crmask=cosmic_clean(dat,sigclip=5.0,sigfrac=0.4,objlim=5.0)
    
  #  datcl[np.where(crmask==True)]=np.nan
    
  #  datred=interpolate_replace_nans(datcl,kern)
   # crmask=(dat<=0.) | (dat >= 25000.)
    # divide by exposure time and append
    #dat/=h.header['elaptime']
    skydat.append( dat)

mjdsky=np.array(mjdsky)  
skycomb=np.mean(skydat,0)#.mean(axis=0)
#skypix=np.sum(1-skymask).astype(float)
#skycomb/=2.0
#skycomb=np.mean(skydat,axis=0)

fits.writeto("spec/Sky_avg.fits",data=skycomb,header=h.header,overwrite=True)
# fits.writeto("spec/Sky_rot.fits",data=np.rot90(skycomb.data,-1),header=h.header,overwrite=True)


# In[10]:


#sky=fits.getdata('spec/Sky_avg.fits')
thresh=3.0
dat=fits.getdata('spec/Sky_avg.fits')
flat=fits.getdata('spec/cal/Nflat.fits')
#datfits.getdata("spec/cal/NFlat.fits")
#sky/=flat
spec,wave=rectify(dat/flat,'spec/spat.map','spec/spec.map')
#dat=polybkg(dat)
#dat/=flat
plt.figure()
ax1=plt.subplot(211)
zlim=ZScaleInterval().get_limits(datcorr)
ax1.imshow(datcorr,origin='lower',vmin=zlim[0],vmax=zlim[1])
datcorr=fringe2D(spec,medwidth=41,thresh_peak=thresh,correct_each_row=False)
plt.show()

datcorr2=fringe2D(spec,medwidth=41,thresh_peak=thresh,correct_each_row=True)
plt.figure()
ax2=plt.subplot(212)
zlim=ZScaleInterval().get_limits(datcorr2)
ax2.imshow(datcorr2,origin='lower',vmin=zlim[0],vmax=zlim[1])
plt.show()


plt.figure()
plt.plot(spec[30:-30,:].mean(0),'k-',alpha=0.5,linewidth=1)
plt.plot(datcorr[30:-30,:].mean(0),'k-',linewidth=2)
plt.plot(datcorr2[30:-30,:].mean(0),'k-',linewidth=2)
plt.xlim(50,1000)
plt.show()


# In[49]:


from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import noise_region_uncertainty,snr_threshold
from astropy.convolution import convolve,Box2DKernel
#from specutils.manipulation import (box_smooth, gaussian_smooth, trapezoid_smooth)
from astropy.constants import c
# ~~ reduce and combine flat

#hobj=[h for h in objic.hdus()]
mjdsky=np.array([Time(h.header['date-obs']+'T'+h.header['utstart'],format='isot',scale='utc').mjd for h in skyic.hdus()])

flat='spec/cal/NFlat.fits'
#sky='spec/Sky_avg.fits'


for obj in objic.files:
   # print("Reducing image: ",h.header['ofname'])
   # hcopy=h.copy()
    #print(h.header['airmass'])
    h=fits.getheader(obj)
    mjdobj=Time(h['date-obs']+'T'+h['utstart'],format='isot',scale='utc').mjd
    isky=np.argmin(np.abs(mjdobj-mjdsky))
    sky=skyic.files[isky]
    print("Sky frame closest in time: ",sky)
    # divide by exposure time
   # hcopy.data/=float(hcopy.header['elaptime'])
    
    # reduce+rectify
    hred=reduce_obj(obj,sky,flat,clean=True,flatten=True,outname='spec/s%04d'%h['frameno'],                    rectify_image=True,bkg_subtract=True,bkg_boxsize=(1,6),fringe_corr=True,filt_width=61,                    fringe_thresh=2.0,specmap='spec/spec.map',spatmap='spec/spat.map')
  

#     fits.writeto('spec/'+hred.header['ofname'][:-5]+".fringe.fits",data=datcorr,\
#                  header=hred.header,overwrite=True)
#     swcs=WCS(hred).sub(0)
#     xpix=np.arange(hred.data.shape[1])
#     wvlgrid,junk=swcs.wcs_pix2world(xpix,np.ones(xpix.size),0)
#     #print(wvlgrid)
#     #wavearr=hred.header['crval1']+hred.header['cdelt1']*
#     #noise_region = SpectralRegion([(10*u.pix,100*u.pix), (900*u.pix, 1600*u.pix)])
#     specred = Spectrum1D(flux=hred.data*u.adu/u.s,spectral_axis=(u.m*wvlgrid).to(u.um),\
#                          rest_value=4.05226*u.um,velocity_convention="optical")
#     specfringe=  Spectrum1D(flux=datcorr*u.adu/u.s,spectral_axis=(u.m*wvlgrid).to(u.um),\
#                          rest_value=4.05226*u.um,velocity_convention="optical")
#     #specunc = noise_region_uncertainty(specred, noise_region)
#     #specred.uncertainty=specunc.uncertainty
#     #print(specred.uncertainty)
    
#     # define observed wavelength, convert to heliocentric frame
#     waveobs=specred.wavelength
#     velobs=specred.velocity
#     #print(waveobs,velobs)
#     vcorr=heliocorr('NGC253',tobs,observatory='Keck')
#     print("Heliocentric velocity correction: ",vcorr)
    
#     wvlhelio = waveobs.to(u.um) * (1. * u.dimensionless_unscaled + vcorr/c.to(u.km/u.s)) 
#     vhelio = velobs.to(u.km/u.s) + vcorr
    
#     #kernel=Box2DKernel(6)
#    # kernel=Gaussian2DKernel(x_stddev=1.0,y_stddev=3.0)
#     #specsmooth = signal.medfilt(specred.flux,[5,7])
#     #specsmooth = convolve(specred, kernel, boundary='fill', fill_value=0., nan_treatment='interpolate')
    
    
#     #specmasked = snr_threshold(specred, 1.5) 
#     #fluxmasked=np.ma.masked_array(specmasked.flux,mask=specmasked.mask,fill_value=np.nan)
#     fig=plt.figure()
#     #zlim=ZScaleInterval().get_limits(fluxmasked)
# #     spatpeak=np.argmax(np.median(datfringe[:,620:740],1))
# #     #spatpeak=133
# #    # print("Peak of spat prof: ",spatpeak)
# #     yex_lo=spatpeak-100
# #     yex_hi=spatpeak+100
#     ax1=fig.add_subplot(211)
#     zlim=ZScaleInterval().get_limits(specred.flux)
#     ax1.imshow(specred.flux,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])

   
#     ax2=fig.add_subplot(212)
#     zlim=ZScaleInterval().get_limits(specfringe.flux)
#     ax2.imshow(specfringe.flux,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])
#     plt.show()
  
    #fits.writeto('spec/'+h.header['ofname'][:-5]+".ff.fits",data=hred.data/flatrect,header=hred.header,overwrite=True)

    


# ## Convert wavelengths/velocities to heliocentric frame

# In[ ]:


#image=fits.getdata("spec/"+"nspec190814_0055.red.fits")
image=fits.getdata("spec/"+"Flatn.rect.fits")
zlim=ZScaleInterval().get_limits(image)
plt.imshow(image,origin='lower',cmap='jet',vmin=zlim[0],vmax=zlim[1])


# In[ ]:


hduflatraw=fits.open(reddir+caldir+"spec/Flatsub.raw.fits")[0]
flatrect,wavearr=rectify(hduflatraw.data,'spat.map','spec.map',mask=hduflatraw.data==1,rotate=True)
normmask=read_ds9("statbox.reg")[0].to_mask().to_image(hdurect.data.shape)
flatnorm,normconst=normalize(hdurect.data,normmask)
print("Normalization constant",normconst)
fits.writeto(reddir+caldir+"spec/Flatnorm.fits",data=flatnorm,header=hdurect.header,overwrite=True)
# flatscaled=ccdp.combine(ccdlist,method='median',scale=1./np.array(medorder),
#                      # sigma_clip=True,sigma_clip_func=np.ma.median, sigma_clip_dev_func=np.ma.std, sigma=4.0,\
#                      minmax_clip=True, minmax_clip_min=0., minmax_clip_max=25000.) 

# flatsnorm,cnorm=normalize(flatscaled.data,normmask)   
# fits.writeto(reddir+caldir+"spec/Flatmed.scaled.fits",data=flatsnorm,header=flatscaled.header,overwrite=True) 


# ### Reduce standard star, check flat-fielding and properties of spectrum. Flatten before or after rectification?

# In[ ]:


# read hdus, select ABBA pattern,
print(std1ic.summary['file'],std1ic.summary['elaptime'])
print(std2ic.summary['file'],std2ic.summary['elaptime'])

flat=fits.getdata(reddir+caldir+"spec/Flatnorm.raw.fits")

# Reduce star 1

hdus=[h for h in std1ic.hdus()][1:] # first exposure is bad
nodAB1=(hdus[0],hdus[1])
nodAB2=(hdus[3],hdus[2])

hduAB1=reduce_objframe(nodAB1,flat)
hduAB2=reduce_objframe(nodAB2,flat)
hduAB1.writeto("hd225200.1.fits")
hduAB2.writeto("hd225200.2.fits")
datAB=np.mean([hduAB1.data,hduAB2.data],axis=0)
fits.writeto("hd225200.fits",data=datAB,header=hduAB1.header,overwrite=True)

# Repeat, for star 2
hdus=[h for h in std2ic.hdus()][1:] # first exposure is bad
nodAB1=(hdus[0],hdus[1])
nodAB2=(hdus[3],hdus[2])

hduAB1=reduce_objframe(nodAB1,flat)
hduAB2=reduce_objframe(nodAB2,flat)
hduAB1.writeto("hd3604.1.fits")
hduAB2.writeto("hd3604.2.fits")
datAB=np.mean([hduAB1.data,hduAB2.data],axis=0)
fits.writeto("hd3604.fits",data=datAB,header=hduAB1.header,overwrite=True)




# In[ ]:


std1=fits.getdata("hd225200.fits")
std2=fits.getdata("hd3604.fits")

zs=ZScaleInterval()
zlim=zs.get_limits(std1.data)
plt.imshow(std1,origin='lower',cmap='viridis',vmin=zlim[0],vmax=zlim[1])
plt.show()

zs=ZScaleInterval()
zlim=zs.get_limits(std2.data)
plt.imshow(std2,origin='lower',cmap='viridis',vmin=zlim[0],vmax=zlim[1])
plt.show()


# ### Reduce object frames. First create master sky, then reduce each object frame individually.

# In[ ]:


hsky=skyic.hdus()
ccdlist=[]
flat = fits.getdata(reddir+caldir+"spec/Flatnorm.raw.fits")
#dark = fits.getdata(reddir+caldir+"spec/Darkmed.scaled.fits")
for hdu in hsky:
    
    fname=hdu.header['ofname']
    print(fname,hdu.header['object'])
    
    hduraw=hdu.copy()
    #print(type(hduraw))
    hdured=reduce_objframe(hduraw,flatdata=flat,cr_clean=True,flatten=False,rectify_image=False)
    fits.writeto(reddir+objdir+'spec/'+fname[:-5]+".raw.fits",data=hdured.data,header=hdured.header,overwrite=True)

    ccd=CCDData(hdured.data,header=hdured.header,unit='adu')
    
    ccdlist.append(ccd)

# combine sky
skycomb=ccdp.combine(ccdlist,method='average',                     minmax_clip=True, minmax_clip_min=0., minmax_clip_max=25000.)
fits.writeto(reddir+objdir+"spec/Sky.raw.fits",data=skycomb.data,header=skycomb.header,overwrite=True)


# In[ ]:


hobj=objic.hdus()
#flatraw = fits.getdata(reddir+caldir+"spec/Flatnorm.raw.fits")
flatraw = fits.getdata(reddir+caldir+"spec/Flatnorm.raw.fits")
flat = fits.getdata(reddir+caldir+"spec/Flatsub.fits")
hdusky = fits.open(reddir+objdir+"spec/nspec190814_0057.raw.fits")[0]

ccdlist=[]
#dark = fits.getdata(reddir+caldir+"spec/Darkmed.scaled.fits")
fig,ax=plt.subplots(10,2)
iplot=0
for hdu in hobj:
    
    fname=hdu.header['ofname']
    print(fname,hdu.header['object'],hdu.header['elaptime'])
    
    hduraw=hdu.copy()
    tobj=hduraw.header['elaptime']
    tsky=hdusky.header['elaptime']

    sky=hdusky.data * (tobj/tsky)
    
    reduced=hduraw.data-sky
    
    #reduced=datasub/flatraw
    
    dataout,wavearr=rectify(reduced,"spat.map","spec.map",rotate=True)
    
    dataout/=flat
    
    zs=ZScaleInterval()
    zlim=zs.get_limits(dataout)
    axi=ax.reshape(ax.size)[iplot]
    axi.imshow(dataout,origin='lower',cmap='cubehelix',vmin=zlim[0],vmax=zlim[1])
    axi.set_xticklabels([])
    axi.set_yticklabels([])
    iplot+=1
    #hduin=
    #hdured=reduce_objframe((hduraw,hdusky),flatdata=flatraw,cr_clean=True,flatten=True,rectify_image=True)
    #fits.writeto(reddir+objdir+'spec/'+fname[:-5]+".red.fits",data=hdured.data,header=hdured.header,overwrite=True)
plt.subplots_adjust(hspace=0.01,wspace=0.01)
plt.show()

#     ccd=CCDData(hdured.data,header=hdured.header,unit='adu')
    
#     ccdlist.append(ccd)

# # combine sky
# objcomb=ccdp.combine(ccdlist,method='sum',\
#                      minmax_clip=True, minmax_clip_min=0., minmax_clip_max=25000.)
# fits.writeto(reddir+objdir+"spec/n253_sum.fits",data=objcomb.data,header=objcomb.header,overwrite=True)


# In[ ]:





# In[ ]:


darkric=ccdp.ImageFileCollection(reddir+caldir+'/spec/darks/')
flatric=ccdp.ImageFileCollection(reddir+caldir+'/spec/flats/')
darks = reduced_images.summary['imagetyp'] == 'DARK'
dark_times = set(darkric.summary['exptime'][])
#print(dark_times)


# In[ ]:





# In[ ]:


from pyraf import iraf
from iraf import noao

noao.imred.ccdred.run()
noao.imred.ccdred.ccdproc.unlearn()
#iraf.ccdred.ccdproc.dParam(cl=0)
#noao.imred.ccdred.ccdproc.dParam(cl=0)
noao.imred.ccdred.ccdproc.biassec=overscanreg
noao.imred.ccdred.ccdproc.trimsec=trimsec
#noao.imred.ccdred.ccdproc.ccdsec=trimsec
noao.imred.ccdred.ccdproc.overscan='yes'
noao.imred.ccdred.ccdproc.trim='yes'
noao.imred.ccdred.ccdproc.zero=''#biasfiles[0]
noao.imred.ccdred.ccdproc.zerocor='no'
noao.imred.ccdred.ccdproc.darkcor='no'
noao.imred.ccdred.ccdproc.flatcor='no'
noao.imred.ccdred.ccdproc.illumcor='no'
noao.imred.ccdred.ccdproc.fringecor='no'
noao.imred.ccdred.ccdproc.fixpix='yes'
noao.imred.ccdred.ccdproc.dark=''#reddir+caldir+'spec/'+'Dark.1.fits'
noao.imred.ccdred.ccdproc.flat=''#reddir+caldir+'spec/'+'Flat.1.fits'
noao.imred.ccdred.ccdproc.interactive='yes'

#noao.imred.ccdred.ccdproc.eParam()


# __Edit image headers to include EXPTIME as the total elapsed integration time__

# ### Run `ccdred.darkcombine` & `ccdred.flatcombine` to create combined dark/flat exposures

# In[ ]:


iraf.imhead('Flat.1.fits',long='yes')
#ccd.ccdmask(reddir+caldir+'spec/Dark.1.fits')


# In[ ]:





# In[ ]:





# In[ ]:


imred=iraf.noao.imred
print(fflist[3:])


# #### Subtract dark from flat

# In[ ]:


hf=fits.open('./cal/spec/Flat.2.fits')
hd=fits.open('./cal/spec/Dark.1.fits')

datfsub=(hf[0].data/float(hf[0].header['exptime']))-(hd[0].data/float(hd[0].header['exptime']))

fits.writeto("./cal/spec/Flat.2.sub.fits",data=datfsub,header=hf[0].header,overwrite=True)

#Flat.3.fits','-','/cal/spec/Dark.1.fits','cal/spec/Flat.3.sub.fits')


# Combine flat frames to make median flat image, & subtract the median dark from that image. __Make sure flats & darks have same exp time!__

# In[ ]:


flathdr = fits.open(flatframes[-1])[0].header
if flathdr['ELAPTIME'] != darkhdr['ELAPTIME']:
    print("ERROR: flats & darks have different exposure times!!!")
flatims = [fits.open( ff )[0].data for ff in flatframes]   
flatcomb = np.nanmedian( np.array(flatims), axis=0)
fits.writeto(outdir+"cal/"+"flat_10s_med.fits",data=flatcomb,header=flathdr,overwrite=True)
# OPTIONAL, SUBTRACT DARK FROM FLAT
#flatmaster = flatcomb - darkcomb
#flatmaster[np.where(flatmaster <=0.)]=1.


# In[ ]:


# Only using one arc image with one dark for now
flatcomb = fits.open(outdir+"cal/"+"flat_10s_med.fits")[0].data
darkcomb = fits.open(outdir+"cal/"+"dark_10s_med.fits")[0].data
get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib notebook

flatsub = flatcomb - darkcomb
fits.writeto(outdir+"cal/"+"flat_10s_sub.fits",data=flatsub,header=flathdr,overwrite=True)
#fig=plt.figure(figsize=(6,6))
#%matplotlib notebook
plt.plot(np.arange(flatcomb.shape[0]),np.nanmedian(flatsub,axis=1))
plt.show()


# In[ ]:


norm_flat = 14000.0 # approximate median of flat in spectral order for normalization


# In[ ]:


flatsub = fits.open(outdir+"cal/"+"flat_10s_sub.fits")[0].data
archdr = fits.open(arcframes[-1])[0].header
adarkhdr = fits.open(adarkframes[-1])[0].header
arc=np.nanmedian([fits.open(af)[0].data for af in arcframes],axis=0)
adark=np.nanmedian([fits.open(adf)[0].data for adf in adarkframes],axis=0)
arcsub = arc - adark
fits.writeto(outdir+"cal/"+"arc_150s_sub.fits",data=arcsub,header=archdr,overwrite=True)


arcflat = arcsub.copy()

arcflat[np.where(flatsub>0.)] /= (flatsub[np.where(flatsub>0.)]/norm_flat)
fits.writeto(outdir+"cal/"+"arc_150s_flattened.fits",data=arcflat,header=archdr,overwrite=True)

#fits.writeto(outdir+"cal/"+"arc_10s_flattened.fits",data=arcflat,header=archdr,overwrite=True)


# __Calibration star__

# Clean cosmic rays using ``astroscrappy.detect_cosmics``

# In[ ]:


stdstar = [fits.getdata(sf) for sf in stdframes]
#print([fits.getheader(sf)['ELAPTIME'] for sf in stdframes])

# clean cosmic rays from raw frames
stdcr=[]
for i in stdstar:
    crmask,cleanim = detect_cosmics(i, sigclip=4.0,             objlim=20, gain=2.85, readnoise=10., niter=4,                                cleantype='medmask', verbose=True)
    stdcr.append(np.ma.masked_array(cleanim,mask=crmask))
stdstar=[s.data for s in stdcr]
#print(stdcr[0].data.shape)  


# In[ ]:


fig=plt.figure(figsize=(15,10))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
imshow_norm(stdstar[0].data,ax1,origin='lower',                    interval=ZScaleInterval(),stretch=SqrtStretch())
imshow_norm(stdcr[0].data,ax2,origin='lower',                    interval=ZScaleInterval(),stretch=SqrtStretch())
plt.show()


# In[ ]:


stdA = np.nanmean([stdstar[0],stdstar[3]],axis=0)
stdB = np.nanmean([stdstar[1],stdstar[2]],axis=0)
stdBneg = stdB*-1.0
# stdAB1 = stdstar[0]-stdstar[1]
# #stdAB1[np.where(flatsub>0.)]/=(flatsub[np.where(flatsub>0.)]/norm_flat) 
# stdAB2 = stdstar[3]-stdstar[2]

fig,axs=plt.subplots(1,2,figsize=(15,10))
axs[0].imshow(stdA-stdB,origin='lower',vmin=-150,vmax=150)
axs[1].imshow(stdA+stdBneg,origin='lower',vmin=-150,vmax=150)
plt.show()
#stdAB2[np.where(flatsub>0.)]/=(flatsub[np.where(flatsub>0.)]/norm_flat) 
stdAB = stdA-stdB #np.nanmean([stdAB1,stdAB2],axis=0)
stdAB[np.where(flatsub>0.)]/=(flatsub[np.where(flatsub>0.)]/norm_flat) 
stdA[np.where(flatsub>0.)]/=(flatsub[np.where(flatsub>0.)]/norm_flat)
stdB[np.where(flatsub>0.)]/=(flatsub[np.where(flatsub>0.)]/norm_flat)
stdhdr = fits.getheader(stdframes[-1])

# flat-field?
fits.writeto(outdir+"cal/"+"std_A_flattened.fits",data=stdA,header=stdhdr,overwrite=True)
fits.writeto(outdir+"cal/"+"std_B_flattened.fits",data=stdB,header=stdhdr,overwrite=True)
#fits.writeto(outdir+"cal/"+"std_Bneg.fits",data=stdBneg,header=stdhdr,overwrite=True)
#fits.writeto(outdir+"cal/"+"std_AB.fits",data=stdAB,header=stdhdr,overwrite=True)
#fits.writeto(outdir+"cal/"+"std_A-2B.fits",data=stdAB+stdBneg,header=stdhdr,overwrite=True)


# # Science frame reduction (either rectified or raw)

# __First read sky spectra, clean of cosmic rays__ 

# In[ ]:


sky=[fits.getdata(skf) for skf in skyframes]
skyhdr=[fits.getheader(skf) for skf in skyframes]


# In[ ]:


skycr=[]
for i in sky:
    crmask,cleanim = detect_cosmics(i, sigclip=4.0,             objlim=20, gain=2.85, readnoise=10., niter=4,                                cleantype='meanmask', verbose=True)
    skycr.append(np.ma.masked_array(cleanim,mask=crmask))
sky=[s for s in skycr]


# In[ ]:


# fig=plt.figure(figsize=(8,8))
# ax1=fig.add_subplot(111)
# #ax2=fig.add_subplot(122)
# imshow_norm(sky[0],ax1,origin='lower',\
#                     interval=ZScaleInterval(),stretch=SqrtStretch())
# # imshow_norm(stdcr[0].data,ax2,origin='lower',\
# #                     interval=ZScaleInterval(),stretch=SqrtStretch())
# plt.show()


# __Create mean sky image__

# In[ ]:


skyavg=np.nanmean(sky,axis=0)
fits.writeto(outdir+"sky_mean.fits",data=skyavg,header=skyhdr[-1],overwrite=True)


# __Read object frames, clean of cosmic rays, subtract sky__

# In[ ]:



for of in objframes:
    sci=fits.getdata(of)
    scihdr=fits.getheader(of)
    frameno=scihdr['framenum']
   # print('%04d'%frameno)
   # break
    
    # clean CRs
    crmask,cleanim = detect_cosmics(sci, sigclip=5.0,             objlim=30, gain=2.85, readnoise=10., niter=4,                    cleantype='meanmask', verbose=True)
    
    # subtract sky
    scisub = cleanim - sky[0].data
    
    #scisub -= np.ma.median(scisub)
    
    # flat-field
    scisub[np.where(flatsub>0.)] /= (flatsub[np.where(flatsub>0.)]/norm_flat)
    
    fits.writeto(outdir+'ns_%04d.fits'%frameno,data=scisub,header=scihdr,overwrite=True)
    


# In[ ]:





# ### Display flat, clip out order for statistics, & normalize. Use ginga Jupyter widget for display.

# # create a Jupyter image that will be our display surface
# # format can be 'jpeg' or 'png'; specify width & height to set viewer size
# # PNG will be a little clearer, especially with overlaid graphics, but
# # JPEG is faster to update
# from IPython import display
# import ipywidgets as widgets
# jup_img = widgets.Image(format='jpeg', width=500, height=500)
# #
# # from ginga.web.pgw import ipg
# # use_opencv = True
# # server = ipg.make_server(host='localhost',numthreads=5,port=9914, use_opencv=use_opencv)
# # #server.thread_pool.stopall()
# # server.start(no_ioloop=True)
# # v1 = server.get_viewer('v1')
# # v1.set_html5_canvas_format('jpeg')

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Boilerplate to create a Ginga viewer connected to this widget
# this could be simplified, or hidden behind a class or convenience 
# method
# NOTE: you don't have to specify a log file--& if you are not interested
# in the log just specify null=True for a null logger
# level=10 will give you the most debugging information
from ginga.misc.log import get_logger
logger = get_logger("my viewer", log_stderr=False, log_file='/tmp/ginga.log', level=40)

from ginga.web.jupyterw.ImageViewJpw import EnhancedCanvasView
v1 = EnhancedCanvasView(logger=logger)
# set our linkage between the jupyter widget at ginga
v1.set_widget(jup_img)

# enable all possible keyboard & pointer operations
bd = v1.get_bindings()
bd.enable_all(True)


# # coordinates = widgets.HTML('<h3>coordinates show up here</h3>')
# # # callback to display position in RA/DEC deg
# # def mouse_move(viewer, button, data_x, data_y, w):
# #     image = viewer.get_image()
# #     if image is not None:
# #         ra, dec = image.pixtoradec(data_x, data_y)
# #         w.value = "x=%f, y=%f"% (ra, dec)
# 
# # v1.add_callback('cursor-changed', mouse_move, coordinates)

# In[ ]:


get_ipython().run_cell_magic('javascript', '', ' /* some magic to keep the cell contents from scrolling\n    (when we embed the viewer)\n  */\nIPython.OutputArea.prototype._should_scroll = function(lines) {\n     return false;\n }')


# In[ ]:


# embed the viewer here

widgets.VBox([jup_img])#,coordinates])#, coordinates])


# In[ ]:


v1.load("flat_master.fits")
# also load as fits hdu


# In[ ]:


# swap XY, flip Y, change colormap back to "ramp"
v1.set_color_map('ramp')
# Set color distribution algorithm
# choices: linear, log, power, sqrt, squared, asinh, sinh, histeq, 
v1.set_color_algorithm('sqrt')
#v1.transform(False, True, True)
v1.rotate(90.)

v1.auto_levels()
v1.show()


# In[ ]:


# add a canvas to the image & set the draw type
canvas = v1.add_canvas()
canvas.set_drawtype('box', color='magenta', fill=True, fillcolor='magenta', fillalpha=0.2)


# In[ ]:


#v1.show()


# In[ ]:


# put the canvas in edit mode
canvas.enable_edit(True)
canvas.set_draw_mode('edit')


# In[ ]:


v1.show()


# In[ ]:


rectorder=canvas.objects[0]


# In[ ]:


img=v1.get_image()
#print(img)
orderimg = img.cutout_shape(rectorder)
normorder = np.ma.median(orderimg)
print(np.median(normorder.data))
#np.ma.median(orderimg)
#print(np.median(orderimg.data),np.ma.median(orderimg))


# __Normalize flat by median & updated hdu__

# In[ ]:


#print(normorder)
hdumast = fits.open("flat_master.fits",mode='update')
ixnz=np.where(hdumast[0].data>0.)
hdumast[0].data[ixnz]/=normorder
#ixnz=np.where(hdumast[0].data>0.)
hdumast.flush()
#print(normorder)
#flatmaster=flatmaster/normorder


# ## Step 2) Create "master arc": dark-subtracted, flat-fielded if necessary

# In[ ]:


# #%matplotlib notebook
# v1.center_image()
# v1.load_data(flatmaster)
# # swap XY, flip Y, change colormap back to "ramp"
# v1.set_color_map('ramp')
# # Set color distribution algorithm
# # choices: linear, log, power, sqrt, squared, asinh, sinh, histeq, 
# v1.set_color_algorithm('sqrt')
# #v1.transform(False, True, True)
# v1.rotate(90.)

# v1.auto_levels()
# v1.show()


# In[ ]:


v1.load("arc_master.fits")

# swap XY, flip Y, change colormap back to "ramp"
v1.set_color_map('smooth')
# Set color distribution algorithm
# choices: linear, log, power, sqrt, squared, asinh, sinh, histeq, 
v1.set_color_algorithm('squared')
#v1.transform(False, True, True)
v1.rotate(90.)

v1.auto_levels()
v1.show()


# ## Step 1) Reduce sky frames

# In[ ]:


flat=fits.open(flatmaster)[0]

# sky_imcl = ImageFileCollection(filenames=skyframes,keywords=['ofname','framenum','elaptime','airmass'])
# sky_hdus=[]
# for i in sky_imcl.hdus():
#     c=CCDData(i.data,unit=u.adu,header=i.header)
#     ccr=cosmicray_lacosmic(c,sigclip=4.0)
#   #  csub=cflat.subtract(d)
#     #cflat=ccr.divide(flat)
#     fits.writeto("nspec_sky_%04d.fits"%i.header['framenum'],data=ccr.data,header=i.header,overwrite=True)
#     sky_hdus.append(fits.PrimaryHDU(ccr.data,i.header))
# # try dividing by flat-field

# am_sky=[h.header['AIRMASS'] for h in sky_hdus]
    
#combiner = Combiner(ccdlist)
#darkflatcomb = combiner.median_combine()    


# 

# In[ ]:


sci_imcl = ImageFileCollection(filenames=sciframes,keywords=['ofname','framenum','elaptime','airmass'])
ccdlist=[]
for i in sci_imcl.hdus():
    
    c=CCDData(i.data,unit=u.adu,header=i.header)
    
    # clean cosmic rays
    ccr=cosmicray_lacosmic(c,sigclip=4.0)
    
    # calculate which sky to use
    am_diff = np.abs(i.header['AIRMASS'] - np.array(am_sky))
    
    ix_sky = np.argmin( am_diff)
    print(sky_hdus[ix_sky])
    # subtract sky
    csub=ccr.subtract(sky_hdus[ix_sky].data*u.electron)
    
    # divide by flat
    cflat = csub.divide(flat.data)
    
    # save reduced fits image
    fits.writeto("nspec_n253_ff_%04d.fits"%i.header['framenum'],data=cflat.data,header=i.header,overwrite=True)
    
    # Calculate which sky image to use (which is closer in time of observation or airmass?)
    
    
#     c.subtract()
#     ccdlist.append(c)
# combiner=Combiner(ccdlist)
# flatmedian = combiner.median_combine()
# ccdlist=[CCDData(i.data,unit=u.adu) for i in flat_imcl.hdus()]
# combiner = Combiner(ccdlist)
# darkflatcomb = combiner.median_combine()  
# %pylab
# fig,ax=plt.subplots(1,1)
# ax.imshow(flatcomb.data,origin='lower')


# ### Use Ginga to interactively select echelle order within flat image, for normalization

# In[ ]:


fits.writeto("arcmaster.fits",data=arcmaster,overwrite=True)


# In[ ]:


v1.load("arcmaster.fits")


# In[ ]:


# Embed the viewer
# swap XY, flip Y, change colormap back to "ramp"
v1.set_color_map('smooth')
# Set color distribution algorithm
# choices: linear, log, power, sqrt, squared, asinh, sinh, histeq, 
v1.set_color_algorithm('histeq')
v1.transform(False, True, True)
v1.auto_levels()
v1.show()


# In[ ]:


# Plot the cuts that we will draw interactively
canvas.delete_all_objects()
canvas.set_drawtype('line')


# In[ ]:


v1.show()


# In[ ]:


def getplot(v1):
    l1 = canvas.objects[0]
    img = v1.get_image()
    values = img.get_pixels_on_line(l1.x1, l1.y1, l1.x2, l1.y2)
    
    #plt.cla()
    plt.figure(figsize=(10,14))
    plt.plot(values)
    plt.ylabel('Pixel value')
    plt.show()


# In[ ]:


getplot(v1)


# ##  Identify & rectify slit  

# In[ ]:




