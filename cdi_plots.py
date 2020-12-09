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
        im = ax.imshow(out.probe.DM_cmd_cycle[ix], interpolation='none', origin='lower',
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
