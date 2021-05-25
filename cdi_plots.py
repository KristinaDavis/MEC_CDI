"""
cdi_plots
Kristina Davis
12/8/2020

Module contains all plotting routines for MEC CDI handling

"""

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import warnings
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.collections import LineCollection


# #########################################################################################################
# Plotting Probes and Probe Focal Plane Response
# #########################################################################################################

def plot_probe_cycle(out):
    """
    plots one complete 0->2pi cycle of the phase probes applied to the DM (DM coordinates)

    :param out:
    :return:
    """
    if out.ts.n_probes >= 4:
        nrows = 2
        ncols = out.ts.n_probes // 2
        figheight = 6
    else:
        nrows = 1
        ncols = out.ts.n_probes
        figheight = 2

    fig, subplot = plt.subplots(nrows, ncols, figsize=(14, figheight))
    fig.subplots_adjust(left=0.02, hspace=.4, wspace=0.2)

    fig.suptitle('DM Probe Cycle')

    for ax, ix in zip(subplot.flatten(), range(out.ts.n_probes)):
        # im = ax.imshow(self.DM_probe_series[ix], interpolation='none', origin='lower')
        im = ax.imshow(out.probe.DM_cmd_cycle[ix], interpolation='none',# origin='lower',
                       vmin=-out.probe.amp, vmax=out.probe.amp)
        ax.set_title(f"Probe " + r'$\theta$=' + f'{out.ts.phase_cycle[ix] / np.pi:.2f}' + r'$\pi$')

    warnings.simplefilter("ignore", category=UserWarning)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical')  #
    cb.set_label(r'$\mu$m', fontsize=12)


def plot_probe_response_cycle(out):
    """
    plots one complete 0->2pi cycle of the phase probes as seen in the focal plane of MEC

    :param out:
    :return:
    """
    if out.ts.n_probes >= 4:
        nrows = 2
        ncols = out.ts.n_probes // 2
        figheight = 6
    else:
        nrows = 1
        ncols = out.ts.n_probes
        figheight = 2

    fig, subplot = plt.subplots(nrows, ncols, figsize=(14, figheight))
    fig.subplots_adjust(left=0.02, hspace=.4, wspace=0.2)

    fig.suptitle('DM Probe Cycle FP Phase Response')

    for ax, ix in zip(subplot.flatten(), range(out.ts.n_probes)):
        probe_ft = (1 / np.sqrt(2 * np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(out.probe.DM_cmd_cycle[ix])))
        nx = 140
        ny = 146

        fr = interpolate.interp2d(range(probe_ft.shape[0]), range(probe_ft.shape[0]), probe_ft.real, kind='cubic')
        fi = interpolate.interp2d(range(probe_ft.shape[0]), range(probe_ft.shape[0]), probe_ft.imag, kind='cubic')
        fr_interp = fr(np.linspace(0, probe_ft.shape[0], ny), np.linspace(0, probe_ft.shape[0], nx))
        fi_interp = fi(np.linspace(0, probe_ft.shape[0], ny), np.linspace(0, probe_ft.shape[0], nx))

        im = ax.imshow(np.arctan2(fi_interp, fr_interp), interpolation='none', cmap='hsv')
        ax.set_title(f"Probe " + r'$\theta$=' + f'{out.ts.phase_cycle[ix] / np.pi:.2f}' + r'$\pi$')

    warnings.simplefilter("ignore", category=UserWarning)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical')  #
    cb.set_label(r'$\theta$', fontsize=12)


def plot_probe_response(out, ix):
    """
    plots the probe appled to the DM as well as its projected response in the focal plane in both amp/phase and
    real/imag

    :return:
    """
    probe_ft = (1 / np.sqrt(2 * np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(out.probe.DM_cmd_cycle[ix])))
    nx = 140
    ny = 146

    fr = interpolate.interp2d(range(probe_ft.shape[0]), range(probe_ft.shape[0]), probe_ft.real, kind='cubic')
    fi = interpolate.interp2d(range(probe_ft.shape[0]), range(probe_ft.shape[0]), probe_ft.imag, kind='cubic')
    fr_interp = fr(np.linspace(0, probe_ft.shape[0], ny), np.linspace(0, probe_ft.shape[0], nx))
    fi_interp = fi(np.linspace(0, probe_ft.shape[0], ny), np.linspace(0, probe_ft.shape[0], nx))


    fig, ax = plt.subplots(3, 2, figsize=(8, 18))
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()

    fig.suptitle(f"\nProbe Amp = {out.probe.amp}, " + r'$\theta$' + f"={out.ts.phase_cycle[ix] / np.pi:.3f}"
                 + r'$\pi$'+
                 f" \nDimensions {out.probe.width}x{out.probe.height}, spacing={out.probe.spacing}\n"
                 )

    im1 = ax1.imshow(out.probe.DM_cmd_cycle[ix], interpolation='none')
    ax1.set_title(f"Probe on DM")
    ax1.set_xlabel('DM x-coord')
    ax1.set_ylabel('DM y-coord')
    #cb = fig.colorbar(im1, ax=ax1)

    ax2.axis('off')
    #ax2('off')

    im3 = ax3.imshow(np.sqrt(fi_interp**2 + fr_interp**2), interpolation='none')
    ax3.set_title("Focal Plane Amplitude")
    ax3.set_xlabel('MEC x-coord')
    ax3.set_ylabel('MEC y-coord')
    #cb = fig.colorbar(im3, ax=ax3)

    im4 = ax4.imshow(np.arctan2(fi_interp, fr_interp), interpolation='none', cmap='hsv')
    ax4.set_title("Focal Plane Phase")
    ax4.set_xlabel('MEC x-coord')
    ax4.set_ylabel('MEC y-coord')

    im5 = ax5.imshow(fr_interp, interpolation='none')
    ax5.set_title(f"Real FT of Probe")
    ax5.set_xlabel('MEC x-coord')
    ax5.set_ylabel('MEC y-coord')

    im6 = ax6.imshow(fr_interp, interpolation='none')
    ax6.set_title(f"Imag FT of Probe")
    ax6.set_xlabel('MEC x-coord')
    ax6.set_ylabel('MEC y-coord')
    # plt.show()  #block=False


def plot_quick_coord_check(out, ix):
    """Plots a quick check of the DM probes interpolated onto MEC FP coordinates"""
    probe_ft = (1 / np.sqrt(2 * np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(out.probe.DM_cmd_cycle[ix])))
    nx = 140
    ny = 146

    fr = interpolate.interp2d(range(probe_ft.shape[0]), range(probe_ft.shape[0]), probe_ft.real, kind='cubic')
    fi = interpolate.interp2d(range(probe_ft.shape[0]), range(probe_ft.shape[0]), probe_ft.imag, kind='cubic')
    fr_interp = fr(np.linspace(0, probe_ft.shape[0], ny), np.linspace(0, probe_ft.shape[0], nx))
    fi_interp = fi(np.linspace(0, probe_ft.shape[0], ny), np.linspace(0, probe_ft.shape[0], nx))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    fig.subplots_adjust(wspace=0.5, right=0.85)

    fig.suptitle('Amplitude Interpolated onto MEC coordinates')
    im = ax.imshow(np.sqrt(fr_interp**2 + fi_interp**2), interpolation='none')


# #########################################################################################################
# Plotting CDI post-processing Results
# #########################################################################################################

# ==================
# FFT of Tweeter Plane
# ==================
def plot_tweeter_fft(out):
    fig, subplot = plt.subplots(1, n_pairs, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.5, right=0.85)
    fig.suptitle('FFT of Tweeter DM Plane')

    tweet = extract_plane(cpx_sequence, 'tweeter')  # eliminates astro_body axis [tsteps,wvl,obj,x,y]
    tweeter = np.sum(tweet, axis=(1, 2))
    for ax, ix in zip(subplot.flatten(), range(n_pairs)):
        fft_tweeter = (1 / np.sqrt(2 * np.pi) *
                       np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(tweeter[ix+n_pairs]))))
        intensity_DM_FFT = np.sum(np.abs(fft_tweeter*mask2D)**2)
        print(f'tweeter fft intensity = {intensity_DM_FFT}')
        im = ax.imshow(np.abs(fft_tweeter*mask2D) ** 2,
                       interpolation='none', norm=LogNorm())  # ,
        # vmin=1e-3, vmax=1e-2)
        ax.set_title(f'Probe Phase ' r'$\theta$' f'={cdi.phase_cycle[ix] / np.pi:.2f}' r'$\pi$')

    cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
    cb.set_label('Intensity')

    # ==================
    # Deltas
    # ==================
    fig, subplot = plt.subplots(1, n_pairs, figsize=(14,5))
    fig.subplots_adjust(wspace=0.5, right=0.85)
    fig.suptitle('Deltas for CDI Probes')

    for ax, ix in zip(subplot.flatten(), range(n_pairs)):
        im = ax.imshow(delta[ix]*1e6*mask2D, interpolation='none',
                       norm=SymLogNorm(linthresh=1),
                       vmin=-1, vmax=1) #, norm=SymLogNorm(linthresh=1e-5))
        ax.set_title(f"Diff Probe\n" + r'$\theta$' + f'={cdi.phase_series[ix]/np.pi:.3f}' +
                     r'$\pi$ -$\theta$' + f'={cdi.phase_series[ix+n_pairs]/np.pi:.3f}' + r'$\pi$')

    cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
    cb.set_label('Intensity')

# ==================
# Original E-Field
# ==================
    fig, subplot = plt.subplots(1, n_nulls, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.5, right=0.85)
    fig.suptitle('Original (Null-Probe) E-field')

    for ax, ix in zip(subplot.flatten(), range(n_nulls)):
        im = ax.imshow(np.abs(fp_seq[n_pairs + ix, 250:270, 150:170]) ** 2,  # , 250:270, 150:170  *mask2D
                       interpolation='none', norm=LogNorm(),
                       vmin=1e-8, vmax=1e-2)
        ax.set_title(f'Null Step {ix}')

    cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
    cb.set_label('Intensity')

# ==================
# E-filed Estimates
# ==================
    fig, subplot = plt.subplots(1, n_nulls, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.5, right=0.85)
    fig.suptitle('Estimated E-field')

    for ax, ix in zip(subplot.flatten(), range(n_nulls)):
        im = ax.imshow(np.abs(E_pupil[ix, 250:270, 150:170])**2,  # , 250:270, 150:170  *mask2D
                       interpolation='none',
                       norm=LogNorm(),
                       vmin=1e-8, vmax=1e-2)  # , norm=SymLogNorm(linthresh=1e-5))
        ax.set_title(f'Null Step {ix}')

    cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
    cb.set_label('Intensity')

# ==================
# Subtracted E-field
# ==================
    fig, subplot = plt.subplots(1, n_nulls, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.5, right=0.85)
    fig.suptitle('Subtracted E-field')

    for ax, ix in zip(subplot.flatten(), range(n_nulls)):
        # im = ax.imshow(np.abs(fp_seq[n_pairs+ix] - np.conj(E_pupil[ix]*mask2D))**2,
        im = ax.imshow(I_processed[ix],  # I_processed[ix, 250:270, 180:200]  I_processed[ix]
                       interpolation='none', norm=SymLogNorm(1e4),  # ,
                       vmin=-1e-6, vmax=1e-6)
        ax.set_title(f'Null Step {ix}')

    cax = fig.add_axes([0.9, 0.2, 0.03, 0.6])  # Add axes for colorbar @ position [left,bottom,width,height]
    cb = fig.colorbar(im, orientation='vertical', cax=cax)  #
    cb.set_label('Intensity')

# ==================
# View Time Series
# ==================
    view_timeseries(cpx_to_intensity(fp_seq*mask2D), cdi, title=f"White Light Timeseries",
                    subplt_cols=sp.tseries_cols,
                    logZ=True,
                    vlim=(1e-7, 1e-4),
                    )


##
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def scale_lD(samp, fn):
    """
    scales the focal plane into lambda/D units. Can use proper.prop_get_fratio to get the f_ratio that proper calculates
    at the focal plane. First convert the sampling in m/pix to rad/pix, then scale by the center wavelength lambda/D
    [rad].

    :param samp: sampling of the wavefront in m/pix
    :param fn: f# (focal ratio) of the beam in the focal plane
    :return:
    """
    wvls = np.linspace(ap.wvl_range[0], ap.wvl_range[1], ap.n_wvl_init)
    cent = np.int(np.floor(ap.n_wvl_final / 2))

    if not samp.shape:
        pass                # sampling is a single value
    else:
        samp = samp[cent]  # sampling at the center wavelength

    # Convert to Angular Sampling Units via platescale
    fl = fn * tp.entrance_d
    rad_scale = samp / fl

    cw = wvls[cent]  # center wavelength
    res = cw / tp.entrance_d

    tic_spacing = np.linspace(0, sp.maskd_size, 5)  # 5 (number of ticks) is set by hand, arbitrarily chosen
    tic_labels = np.round(np.linspace(-rad_scale * sp.maskd_size / 2 , rad_scale * sp.maskd_size / 2 , 5)/res)  # nsteps must be same as tic_spacing
    tic_spacing[0] = tic_spacing[0] + 1  # hack for edge effects
    tic_spacing[-1] = tic_spacing[-1] - 1  # hack for edge effects

    axlabel = (r'$\lambda$' + f'/D')

    return tic_spacing, tic_labels, axlabel


def get_fp_mask(cdi, thresh=1e-7):
    """
    returns a mask of the CDI probe pattern in focal plane coordinates

    :param cdi: structure containing all CDI probe parameters
    :param thresh: intensity threshold for determining probed coordinates
    :return: fp_mask: boolean array where True marks the probed coordinates
             imsk, jmsk:
             irng, jrng:

    """
    nx = sp.grid_size
    ny = sp.grid_size
    dm_act = cdi.nact

    fftA = (1 / np.sqrt(2 * np.pi) *
            np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cdi.DM_probe_series[0]))))

    Ar = interpolate.interp2d(range(dm_act), range(dm_act), fftA.real, kind='cubic')
    Ai = interpolate.interp2d(range(dm_act), range(dm_act), fftA.imag, kind='cubic')
    ArI = Ar(np.linspace(0, dm_act, ny), np.linspace(0, dm_act, nx))
    AiI = Ai(np.linspace(0, dm_act, ny), np.linspace(0, dm_act, nx))

    fp_probe = np.sqrt(ArI**2 + AiI**2)
    # fp_mask = (fp_probe > 1e-7)
    # (imsk, jmsk) = (fp_probe > 1e-7).nonzero()
    fp_mask = (fp_probe > thresh)
    (imsk, jmsk) = (fp_probe > thresh).nonzero()

    irng = range(min(imsk), max(imsk), 1)
    jrng = range(min(jmsk), max(jmsk), 1)

    edges = get_all_edges(bool_img=fp_mask.T)
    edges = edges - 0.5  # convert indices to coordinates; TODO adjust according to image extent
    outlines = close_loop_edges(edges=edges)

    # imx = max(irng)-1  # -1 is to get index values for plotting purposes
    # imn = min(irng)-1
    # jmx = max(jrng)-1
    # jmn = min(jrng)-1

    return fp_mask, outlines, imsk, jmsk, irng, jrng


def get_all_edges(bool_img):
    """
    Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
    The returned array edges has he dimension (n, 2, 2).
    Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
    Note that the indices of a pixel also denote the coordinates of its lower left corner.
    """
    edges = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
            edges.append(np.array([[i, j+1],
                                   [i+1, j+1]]))
        # East
        if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
            edges.append(np.array([[i+1, j],
                                   [i+1, j+1]]))
        # South
        if j == 0 or not bool_img[i, j-1]:
            edges.append(np.array([[i, j],
                                   [i+1, j]]))
        # West
        if i == 0 or not bool_img[i-1, j]:
            edges.append(np.array([[i, j],
                                   [i, j+1]]))

    if not edges:
        return np.zeros((0, 2, 2))
    else:
        return np.array(edges)


def close_loop_edges(edges):
    """
    Combine thee edges defined by 'get_all_edges' to closed loops around objects.
    If there are multiple disconnected objects a list of closed loops is returned.
    Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
    """
    loop_list = []
    while edges.size != 0:

        loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
        edges = np.delete(edges, 0, axis=0)

        while edges.size != 0:
            # Get next edge (=edge with common node)
            ij = np.nonzero((edges == loop[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                loop.append(loop[0])
                # Uncomment to to make the start of the loop invisible when plotting
                # loop.append(loop[1])
                break

            loop.append(edges[i, (j + 1) % 2, :])
            edges = np.delete(edges, i, axis=0)

        loop_list.append(np.array(loop))

    return loop_list
