"""
scexao_model.py
Kristina Davis


"""
import numpy as np
import proper
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from SubaruPupil import SubaruPupil
from errormap import errormap

#################################################################
# SCExAO Optial Properties
# ################################################################
# SCExAO
beam_ratio = 0.5  # 0.3516 gives aperture sampling of .4 mm (the actual dm pitch) at lambda=900 um
entrance_d = 1.2  # size of beam on the DM is 18 mm^2

# SCExAO Optics
fl_SxOAPG = 0.25  # m focal length of Genera SCExAO lens (OAP1,3,4,5)
d_SxOAPG = 0.5  # diameter of SCExAO OAP's are 2 inches=0.051 m
# These distances aren't actually working, so just doing very basic, 4F optical systems until further notice

# ------------------------------
# DM
dm_act = 50  # SCExAO has 50x50 actuators
dm_pitch = 0.004  # [m] pixel pitch of the DM actuators --> sampling of the DM fits file=400 um/pix

#################################################################################################
# SCExAO Model
# ###############################################################################################
def dummy_telescope(lmda, grid_size, kwargs):
    """
    propagates instantaneous complex E-field thru Subaru from the DM through SCExAO

    uses PyPROPER3 to generate the complex E-field at the pupil plane, then propagates it through SCExAO 50x50 DM,
        then coronagraph, to the focal plane
    :returns spectral cube at instantaneous time in the focal_plane()
    """
    # print("Propagating Broadband Wavefront Through Subaru")

    # Initialize the Wavefront in Proper
    wfo = proper.prop_begin(entrance_d, lmda, grid_size, beam_ratio)

    # Defines aperture (baffle-before primary)
    proper.prop_circular_aperture(wfo, entrance_d/2)
    proper.prop_define_entrance(wfo)  # normalizes abs intensity

    # SCExAO Reimaging 1
    proper.prop_lens(wfo, fl_SxOAPG)
    proper.prop_propagate(wfo, fl_SxOAPG * 2)  # move to second pupil

    ########################################
    # Import/Apply Actual DM Map
    # #######################################
    plot_flag = False
    if kwargs['verbose'] and kwargs['ix'] == 0:
        plot_flag = True
    dm_map = kwargs['map']
    # flat = proper.prop_zernikes(wfo, [2, 3], np.array([5, 1]))  # zernike[2,3] = x,y tilt
    # adding a tilt for shits and giggles
    # proper.prop_propagate(wfo, fl_SxOAPG)  # from tweeter-DM to OAP2
    errormap(wfo, dm_map, SAMPLING=dm_pitch, MIRROR_SURFACE=True, MICRONS=True, BR=beam_ratio,
             PLOT=plot_flag)  # WAVEFRONT=True
    # errormap(wfo, dm_map, SAMPLING=dm_pitch, AMPLITUDE=True, BR=beam_ratio, PLOT=plot_flag)  # WAVEFRONT=True
    # proper.prop_circular_aperture(wfo, entrance_d/2)

    if kwargs['verbose'] and kwargs['ix'] == 0:
        fig, subplot = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        ax1, ax2 = subplot.flatten()
        fig.suptitle('SCExAO Model WFO after errormap', fontweight='bold', fontsize=14)
        # ax.imshow(dm_map, interpolation='none')
        ax1.imshow(np.abs(proper.prop_shift_center(wfo.wfarr)) ** 2, interpolation='none')
        ax1.set_title('Amplitude')
        ax2.imshow(np.angle(proper.prop_shift_center(wfo.wfarr)), interpolation='none',
                   vmin=-2 * np.pi, vmax=2 * np.pi)  # , cmap='hsv'
        ax2.set_title('Phase')
    # ------------------------------------------------
    # proper.prop_propagate(wfo, fl_SxOAPG)  # from tweeter-DM to OAP2

    # SCExAO Reimaging 2
    proper.prop_lens(wfo, fl_SxOAPG)
    proper.prop_propagate(wfo, fl_SxOAPG)  # focus at exit of DM telescope system
    proper.prop_lens(wfo, fl_SxOAPG)
    proper.prop_propagate(wfo, fl_SxOAPG)  # focus at exit of DM telescope system
    # ########################################
    # # Focal Plane
    # # #######################################
    # Check Sampling in focal plane
    # shifts wfo from Fourier Space (origin==lower left corner) to object space (origin==center)
    # proper.prop_shift_center(wfo.wfarr)
    # wf, samp = proper.prop_end(wfo, NoAbs=True)
    wf = proper.prop_shift_center(wfo.wfarr)
    samp = proper.prop_get_sampling(wfo)
    # smp_asec = proper.prop_get_sampling_arcsec(wfo)

    if kwargs['verbose'] and kwargs['ix'] == 0:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle('SCExAO Model Focal Plane', fontweight='bold', fontsize=14)
        ax.imshow(np.abs(wf)**2, interpolation='none',
                  norm=LogNorm(vmin=1e-7,vmax=1e-2))  # np.abs(proper.prop_shift_center(wfo.wfarr))**2
    #
    # if kwargs['verbose'] and kwargs['ix']==0:
    #     print(f"\n\tFocal Plane\n"
    #           f"sampling at focal plane is {smp_asec * 1e3:.4f} mas\n"
    #           f"\tfull FOV is {smp_asec * grid_size * 1e3:.2f} mas")
    #     s_rad = proper.prop_get_sampling_radians(wfo)
    #     print(f"sampling at focal plane is {s_rad * 1e6:.6f} urad")

        # print(f"Finished simulation")

    return wf, samp



