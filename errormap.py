#   Copyright 2016, 2017 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the
#   file "LegalStuff.txt" in the PROPER library directory.
#
#   PROPER developed at Jet Propulsion Laboratory/California Inst. Technology
#   Original IDL version by John Krist
#   Python translation by Navtej Saini, with Luis Marchen and Nikta Amiri
import warnings

import proper
import numpy as np

def errormap(wf, dm_map, xshift = 0., yshift = 0., **kwargs):
    """Read in a surface, wavefront, or amplitude error map from a FITS file. 
    
    Map is assumed to be in meters of surface error. One (and only one) of the 
    MIRROR_SURFACE, WAVEFRONT, or AMPLITUDE switches must be specified in order 
    to properly apply the map to the wavefront.  For surface or wavefront error 
    maps, the map values are assumed to be in meters, unless the NM or MICRONS 
    switches are used to specify the units. The amplitude map must range 
    from 0 to 1.  The map will be interpolated to match the current wavefront 
    sampling if necessary.
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
        
    dm_map : 2D numpy array
        the DM map in units of surface deformation
        
    xshify, yshift : float
        Amount to shift map (meters) in X,Y directions
    
    Returns
    -------
    DMAP : numpy ndarray
        Returns map (after any necessary resampling) if set.
    
    
    Other Parameters
    ----------------
    XC_MAP, YC_MAP : float
        Pixel coordinates of map center (Assumed n/2,n/2)
        
    SAMPLING : float
        Sampling of map in meters
        
    ROTATEMAP : float
        Degrees counter-clockwise to rotate map, after any resampling and 
        shifting
        
    MULTIPLY : float
        Multiplies the map by the specified factor
        
    MAGNIFY : float
        Spatially magnify the map by a factor of "constant" from its default
        size; do not use if SAMPLING is specified
        
    MIRROR_SURFACE : bool
        Indicates file contains a mirror surface height error map; It assumes a 
        positive value indicates a surface point higher than the mean surface.  
        The map will be multiplied by -2 to convert it to a wavefront map to 
        account for reflection and wavefront delay (a low region on the surface 
        causes a positive increase in the phase relative to the mean)
        
    WAVEFRONT : bool
        Indicates file contains a wavefront error map
        
    AMPLITUDE : bool
        Indicates file contains an amplitude error map
        
    NM or MICRONS : bool
        Indicates map values are in nanometers or microns. For surface or 
        wavefront maps only
        
    Raises
    ------
    SystemExit:
        If AMPLITUDE and (NM or MICRONS) parameters are input.
        
    SystemExit:
        If NM and MICRONS parameteres are input together. 
        
    ValueError:
        If map type is MIRROR_SURFACE, WAVEFRONT, or AMPLITUDE.
    """
    if ("AMPLITUDE" in kwargs and kwargs["AMPLITUDE"]) \
       and (("NM" in kwargs and kwargs["NM"]) \
       or ("MICRONS" in kwargs and kwargs["MICRONS"])):
        raise SystemExit("ERRORMAP: Cannot specify NM or MICRON for an amplitude map")
    
    if ("NM" in kwargs and kwargs["NM"]) and \
       ("MICRONS" in kwargs and kwargs["MICRONS"]):
        raise SystemExit("ERRORMAP: Cannot specify both NM and MICRONS")

    # KD edit: try to get the dm map to apply only in regions of the beam
    n = proper.prop_get_gridsize(wf)  # should be 128x128
    new_sampling = proper.prop_get_sampling(wf)  #kwargs["SAMPLING"]  #*dm_map.shape[0]/npix_across_beam
    if new_sampling > (kwargs["SAMPLING"] + kwargs["SAMPLING"]*.1) or \
        new_sampling < (kwargs["SAMPLING"] - kwargs["SAMPLING"]*.1):
            print(f'User-defined samping is {kwargs["SAMPLING"]:.6f} but proper wavefront has sampling of '
                  f'{new_sampling:.6f}')
            warnings.warn(f'User-defined beam ratio does not produce aperture sampling consistent with SCExAO actuator '
                          f'spacing. May produce invalid results')

    if not "XC_MAP" in kwargs:
        s = dm_map.shape
        xc = s[0] // 2
        yc = s[1] // 2
    else:
        xc = kwargs["XC_MAP"]
        yc = kwargs["YC_MAP"]

    # resample dm_map to size of beam in the simulation
    # grid = proper.prop_resamplemap(wf, dm_map, new_sampling, xc, yc, xshift, yshift)
    dmap=np.zeros((wf.wfarr.shape[0],wf.wfarr.shape[1]))
    r = dmap.shape
    xrc = r[0] // 2
    yrc = r[1] // 2
    dmap[xrc-xc:xrc+xc, yrc-yc:yrc+yc] = dm_map

    # Create mask to eliminate resampling artifacts outside of beam
    if ("MASKING" in kwargs and kwargs["MASKING"]):
        h, w = wf.wfarr.shape[:2]
        center = (int(w / 2), int(h / 2))
        radius = np.ceil(h * kwargs['BR'] / 2)  #
        # Making the Circular Boolean Mask
        Y, X = np.mgrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        inds = dist_from_center <= radius
        # Applying the Mask to the dm_map
        mask = np.zeros_like(dmap)
        mask[inds] = 1
        dmap *= mask

    # Shift the center of dmap to 0,0
    dmap = proper.prop_shift_center(dmap)

    if kwargs['PLOT']:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, SymLogNorm

        fig, subplot = plt.subplots(nrows=1,ncols=2, figsize=(12, 5))
        ax1, ax2 = subplot.flatten()
        fig.suptitle(f'RTC DM Voltage Maps')

        ax1.imshow(dm_map, norm=SymLogNorm(1e-2))  # LogNorm(vmin=np.min(dm_map),vmax=np.max(dm_map))  SymLogNorm(1e-2)
        ax1.set_title('DM Map Read In')
        ax2.imshow(proper.prop_shift_center(dmap))  #  , cmap='hsv' must shift the center because
        # proper assumes dmap center is 0,0, so we need to shift it back to plot properly
        ax2.set_title('DM Map in Center of Proper simulated beam')

    if "ROTATEMAP" in kwargs or "MAGNIFY" in kwargs:
        # readmap stores map with center at (0,0), so shift
        # before and after rotation
        dmap = proper.prop_shift_center(dmap)
        if "ROTATEMAP" in kwargs:
            dmap = proper.prop_rotate(dmap, kwargs["ROTATEMAP"], CUBIC=-0.5, MISSING=0.0)
        if "MAGNIFY" in kwargs:
            dmap = proper.prop_magnify(dmap, kwargs["MAGNIFY"], dmap.shape[0])
            dmap = proper.prop_shift_center(dmap)
            
    if ("MICRONS" in kwargs and kwargs["MICRONS"]):
        dmap *= 1.e-6
        
    if ("NM" in kwargs and kwargs["NM"]):
        dmap *= 1.e-9
        
    if "MULTIPLY" in kwargs:
        dmap *= kwargs["MULTIPLY"]
        
    i = complex(0.,1.)

    if ("MIRROR_SURFACE" in kwargs and kwargs["MIRROR_SURFACE"]):
        wf.wfarr *= np.exp(-4*np.pi*i/wf.lamda * dmap)  # Krist version
    elif "WAVEFRONT" in kwargs:
        wf.wfarr *= np.exp(2*np.pi*i/wf.lamda * dmap)
    elif "AMPLITUDE" in kwargs:    
        wf.wfarr *= dmap
    else:
        raise ValueError("ERRORMAP: Unspecified map type: Use MIRROR_SURFACE, WAVEFRONT, or AMPLITUDE")

    # check1 = proper.prop_get_sampling(wf)
    # print(f"\n\tErrormap Sampling\n"
    #       f"sampling in errormap.py is {check1 * 1e3:.4f} mm\n")

    return dmap


"""
    
# # Unwrap Phase
    # from skimage.restoration import unwrap_phase
    # amp_map = proper.prop_get_amplitude(wf)
    # phs_map = proper.prop_get_phase(wf)
    # unwrapped = unwrap_phase(phs_map, wrap_around=[False, False])
    # wf.wfarr = proper.prop_shift_center(amp_map * np.cos(unwrapped) + 1j * amp_map * np.sin(unwrapped))
"""