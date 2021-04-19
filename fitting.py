#!/usr/bin/env 
import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')
from optparse import OptionParser
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.interpolate import interp1d
from scipy.ndimage import interpolation as interp
from scipy.optimize import curve_fit
from scipy.stats import skewnorm


""" 
    Functions and scripts to perform basic analysis on a spectrum and spectral lines therein. Includes functions to fit Gaussian profiles 
    and continuum.
    

"""


#~~~ GAUSSIAN LINE MODELS
def gauss(x,*P):
    """
    Single-gaussian model
    Parameters: a -- Amplitude
                b -- Centroid
                c -- Sigma
    Since no baseline, assumes continuum has already been subtracted.
    """

    a,b,c = P
    
    fout = a * np.exp(- 0.5 * (x-b)**2 / c**2 )

    return fout

def gauss2(x,*P):
    """
    Single-gaussian model
    Parameters: a -- Amplitude
                b -- Centroid
                c -- Sigma
    Since no baseline, assumes continuum has already been subtracted.
    """

    a1,b,c1,a2,c2 = P
    
    fout = a1 * np.exp(- 0.5 * (x-b)**2 / c1**2 ) + a2 * np.exp(- 0.5 * (x-b)**2 / c2**2 )

    return fout
    
def gaussn(x,*P):
    """
    Single-gaussian model
    Parameters: {a1,b1,c1,...,an,bn,cn} where:
                ai -- Amplitude of ith component
                bi -- Centroid ' '
                ci -- Sigma ' '
    
    Since no baseline, assumes continuum has already been subtracted.
    """
    
    # print P,len(P)
    n = len(P)//3
  #  print("Num components: ",n)
    fout = np.zeros(x.size)
    for i in range(0,int(n*3),3):
        ai=P[i]
        bi=P[i+1]
        ci=P[i+2]
       # print(ai,bi,ci)
        fout += ai * np.exp(- 0.5 * (x-bi)**2 / ci**2 )
    
    return fout
   
   
def poly(x,*P):
    n = len(P)
    fout=np.zeros(x.size)
    for i in range(n):
        fout += P[i] * x**i
        
    return fout
   
        
def gauss_poly(x,*P):
    """
    Single-gaussian +  nth-order polyonmial 
    Parameters: a -- Amplitude
                b -- Centroid
                c -- Sigma
                p0, p1 -- 0th/1st order poly coefficients poly = p0 + p1 * x
    """

    # a,b,c,p0,p1 = P
    
    fout = gauss(x,*P[:3])  +  poly(x,*P[3:])

    return fout
    
def gaussn_poly0(x,*P):
    """
    Single-gaussian +  nth-order polyonmial 
    Parameters: a -- Amplitude
                b -- Centroid
                c -- Sigma
                p0, p1 -- 0th/1st order poly coefficients poly = p0 
    """

    # a,b,c,p0,p1 = P
    
    fout = gaussn(x,*P[:-1])  +  poly(x,*P[-1:])

    return fout
    
def gaussn_poly1(x,*P):
    """
    Single-gaussian +  nth-order polyonmial 
    Parameters: a -- Amplitude
                b -- Centroid
                c -- Sigma
                p0, p1 -- 0th/1st order poly coefficients poly = p0 + p1 * x
    """

    # a,b,c,p0,p1 = P
    
    fout = gaussn(x,*P[:-2])  +  poly(x,*P[-2:])

    return fout
    
def gaussn_poly2(x,*P):
    """
    Single-gaussian +  nth-order polyonmial 
    Parameters: a -- Amplitude
                b -- Centroid
                c -- Sigma
                p0, p1, p2 -- 0th/1st order poly coefficients poly = p0 + p1 * x + p2 * x**2
    """

    # a,b,c,p0,p1 = P
    
    fout = gaussn(x,*P[:-3])  +  poly(x,*P[-3:])

    return fout
    

#~~~ FUNCTIONS FOR FITTING SPECTRA
#
#
# def fitprop(paropt,parerrs="None",gaussn=2):
#     """
#     From best-fit gaussian parameters to brackett alpha and He I line, derive important kinematic quantities.
#     paropt should be 3 or 6 parameters (a1,x1,sigma1,a2,x2,sigma2) with last 3 elements set to zero if
#     gaussn=1 (if only fit 1-component gaussian).
#
#     Returns: 8-element tuple with following elements:
#           (velocity centroid of 1, err vel centroid 1, fwhm vel 1, err fwhm vel 1, vel centroid 2,
#                 err vel centroid 2, fwhm vel 2, err fwhm vel 2)
#
#
#     """
#     c = 299792.458 # define speed of light in km/s
#     c_ms = c * 1e3
#     w0_bra=4.05226
#     w0_heI=4.04900
#     # nurest = c * 1e3 / (wvlrest * 1e-6)
#
#     # Define velocity conversion functions
#     def vopt(wvlobs,wvlrest,ewvlobs=None):
#         """ Optical definition of velocity
#             wvlobs,ewvlobs in um
#
#          """
#         vout = c * (wvlobs - wvlrest) / wvlrest
#         if ewvlobs == None:
#             return vout
#         else:
#             evout = (c/wvlrest) * ewvlobs
#             return vout,evout
#
#     # def vrad(wvlobs,ewvlobs=None):
#     #     """ Radio def of velocity
#     #         wvlobs,ewvlobs in um
#     #
#     #      """
#     #     nuobs = c_ms / (wvlobs * 1e-6) # in Hz
#     #     vout = c * (nurest - nuobs) / nurest
#     #     if ewvlobs == None:
#     #         return vout
#     #     else:
#     #         enuobs=(c_ms / (wvlobs*1e-6)**2) * ewvlobs*1e-6 # err in Hz
#     #         evout = (c/nurest) * enuobs
#     #         return vout,evout
#
#     def fwhmv(sigmawvl,wvlobs,esigmawvl=None,ewvlobs=None):
#         fwhmwvl = 2. * np.sqrt( 2. * np.log(2.) ) * sigmawvl # in micro n
#         fwhmvel = c * fwhmwvl / wvlobs # in km/s
#         if (esigmawvl == None) | (ewvlobs == None):
#             return fwhmvel
#         else:
#             efwhmwvl= 2. * np.sqrt( 2. * np.log(2.) ) * esigmawvl
#             efwhmvel = c * np.sqrt( (1./wvlobs)**2 * efwhmwvl**2 + (fwhmwvl/wvlobs**2)**2 * ewvlobs**2  )
#             return fwhmvel,efwhmvel
#
#
#     # def vlsr(v,l=314.8603,b=30.1060):
#     #     lrad=l * np.pi/180
#     #     brad=b * np.pi/180
#     #     return v + 9. * np.cos(lrad) * cos(brad) + 12. * np.sin(lrad)* np.cos(brad)+ 7. * np.sin(brad)
#     #
#
#     # now errors if relevant
#     if parerrs!="None": # then calculate/return errors as well
#
#         # One component always present, calculate quantities for that
#         amp1 = paropt[0]
#         wvl1=paropt[1]
#         nu1 = c_ms / (wvl1 * 1e-6) # in Hz
#         sigma1=paropt[2]
#         amp1err = parerrs[0]
#         wvl1err=parerrs[1]
#         sigma1err=parerrs[2]
#
#         # calculate centroid velocities
#         #v1,v1err = vopt(wvl1,ewvlobs=wvl1err)
#         v1,v1err = vopt(wvl1,w0_bra,ewvlobs=wvl1err)
#
#         # FWHM of velocity
#         fwhm1,fwhm1err = fwhmv(sigma1,wvl1,esigmawvl=sigma1err,ewvlobs=wvl1err)
#
#
#         if gaussn==2:
#             # calculate quantities for second component and return all
#             amp2 = paropt[3]
#             wvl2=paropt[4]
#             nu2 = c_ms / (wvl2 * 1e-6) # in Hz
#             sigma2 = paropt[5]
#             amp2err = parerrs[3]
#             wvl2err=parerrs[4]
#             sigma2err=parerrs[5]
#
#             # Centroid vel
#             #v2,v2err= vopt(wvl2,ewvlobs=wvl2err)
#             v2,v2err= vopt(wvl2,w0_heI,ewvlobs=wvl2err)
#
#             # FWHM of velocity
#             fwhm2,fwhm2err = fwhmv(sigma2,wvl2,esigmawvl=sigma2err,ewvlobs=wvl2err)
#
#             # now difference in centroid velocities
#             # dv12 = v1 - v2
#             # dv12err = np.sqrt( v1err**2 + v2err**2 )
#
#             prop= (v1,v1err,fwhm1,fwhm1err,v2,v2err,fwhm2,fwhm2err)#,dv12,dv12err)
#
#         elif gaussn==1:
#
#
#             prop= (v1,v1err,fwhm1,fwhm1err)
#
#     elif parerrs=="None":
#         # calculate and return quantities w/o errs
#         # One component always present, calculate quantities for that
#         amp1 = paropt[0]
#         wvl1=paropt[1]
#         nu1 = c_ms / (wvl1 * 1e-6) # in Hz
#         sigma1=paropt[2]
#
#         # calculate centroid velocity
#         #v1 = vopt(wvl1)
#         v1 = vopt(wvl1,w0_bra)
#
#         # FWHM of velocity
#         fwhm1 = fwhmv(sigma1,wvl1)
#
#
#         if gaussn==2:
#             # calculate quantities for second component and return all
#             amp2 = paropt[3]
#             wvl2=paropt[4]
#             nu2 = c_ms / (wvl2 * 1e-6) # in Hz
#             sigma2 = paropt[5]
#
#             # Centroid vel
#             #v2 = vopt(wvl2)
#             v2 = vopt(wvl2,w0_heI)
#
#             # FWHM of velocity
#             fwhm2 = fwhmv(sigma2,wvl2)
#
#             # now difference in centroid velocities
#             # dv12 = v1 - v2
#
#             prop= (v1,fwhm1,v2,fwhm2)#,dv12)
#
#         elif gaussn==1:
#
#             prop= (v1,fwhm1)
#
#
#     return prop
#
#
#
    