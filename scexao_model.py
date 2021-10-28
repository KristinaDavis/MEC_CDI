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
from cdi_plots import scale_lD, extract_center, get_fp_mask, add_colorbar

#################################################################
# SCExAO Optial Properties
# ################################################################
# SCExAO
beam_ratio = 0.3516  # 0.3516 gives aperture sampling of .4 mm (the actual dm pitch) at lambda=900 um
entrance_d = 0.018  # size of beam on the DM is 18 mm^2

# SCExAO Optics
fl_SxOAPG = 0.255  # m focal length of Genera SCExAO lens (OAP1,3,4,5)
fl_SxOAP2 = 0.519  # m focal length of SCExAO OAP 2
d_SxOAPG = 0.0508  # diameter of SCExAO OAP's are 2 inches=0.051 m
# These distances aren't actually working, so just doing very basic, 4F optical systems until further notice
dist_SxOAP1_scexao = 0.1345  # m
dist_scexao_sl2 = 0.2511 - dist_SxOAP1_scexao  # m
dist_sl2_focus = 0.1261  # m

# ------------------------------
# DM
dm_act = 50  # SCExAO has 50x50 actuators
dm_pitch = 0.0004  # [m] pixel pitch of the DM actuators --> sampling of the DM fits file=400 um/pix

# ------------------------------
# Coronagraph
# cg_size = 1.5  # physical size or lambda/D size
# cg_size_units = "l/D"  # "m" or "l/D"
fl_cg_lens = fl_SxOAPG  # m
lyot_size = 0.95  # units are in fraction of surface un-blocked

# ------------------------------
# MEC Optics
mec_parax_fl = 0.1  # [m] paraxial lens (pickoff lens?) focal length
mec1_fl = 0.1  # [m] optic 1 focal length = 100 mm
mec2_fl = 0.009 # [m] optic 2 focal length = 9 mm
mec3_fl = .3  # [m] optic 3 focal length = 300 mm
# mec_l1_l2 = 0.104916  # distance between lens 1 and lens2 in MEC optics, 104.916 mm as seen on MEC_final.zmx
mec_l1_l2 = mec1_fl + mec2_fl  # mec1_fl+mec2_fl = 109 mm
mec_l2_l3 = 0.32868  # 0.32868 from the zemax model, colimized beam so doesn't matter much how far
mec_l3_focus = mec3_fl  # reading off the zemax file  (11.5/2) + 140.25+12.7+25.8+10+4+20+83.5==302.0 mm
# mec_l1_l2 = 0.0986536  # distance between lens 1 and lens2 in MEC optics, ==3.884 inches as seen on MEC_final.cad
# mec_l2_l3 = (1.544+6.511+4.614)*25.4/1e3 == 321.792 mm  # convert inches to m, reading off CAD model

#################################################################################################
# SCExAO Model
# ###############################################################################################
def scexao_model(lmda, grid_size, kwargs):
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

    # Test Sampling
    if kwargs['verbose'] and kwargs['ix']==0:
        check1 = proper.prop_get_sampling(wfo)
        print(f"\n\tDM Pupil Plane\n"
              f"sampling at aperture is {check1 * 1e3:.4f} mm\n"
              f"Total Sim is {check1 * 1e3 * grid_size:.2f}x{check1 * 1e3 * grid_size:.2f} mm\n"
              f"Diameter of beam is {check1 * 1e3 * grid_size * beam_ratio:.4f} mm over {grid_size * beam_ratio} pix")

    # SCExAO Reimaging 1
    proper.prop_lens(wfo, fl_SxOAPG)  # produces f#14 beam (approx exit beam of AO188)
    proper.prop_propagate(wfo, fl_SxOAPG * 2)  # move to second pupil
    if kwargs['verbose'] and kwargs['ix']==0:
        print(f"initial f# is {proper.prop_get_fratio(wfo):.2f}\n")

    ########################################
    # Import/Apply Actual DM Map
    # #######################################
    plot_flag = False
    if kwargs['verbose'] and kwargs['ix']==0:
        plot_flag=True

    dm_map = kwargs['map']
    errormap(wfo, dm_map, SAMPLING=dm_pitch, MIRROR_SURFACE=True, MASKING=True,
             BR=beam_ratio, PLOT=plot_flag)  # MICRONS=True

    if kwargs['verbose'] and kwargs['ix']==0:
        fig, subplot = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        ax1, ax2 = subplot.flatten()
        fig.suptitle('SCExAO Model WFO after errormap', fontweight='bold', fontsize=14)
        ax1.imshow(proper.prop_get_amplitude(wfo), interpolation='none')  # np.abs(proper.prop_shift_center(wfo.wfarr))**2
        ax1.set_title('Amplitude')
        ax2.imshow(proper.prop_get_phase(wfo), interpolation='none',  # np.angle(proper.prop_shift_center(wfo.wfarr))
                   vmin=-1*np.pi, vmax=1*np.pi, cmap='hsv')  # , cmap='hsv'
        ax2.set_title('Phase')

    # ------------------------------------------------
    # SCExAO Reimaging 2
    proper.prop_lens(wfo, fl_SxOAPG)
    proper.prop_propagate(wfo, fl_SxOAPG)  # focus at exit of DM telescope system
    proper.prop_lens(wfo, fl_SxOAPG)
    proper.prop_propagate(wfo, fl_SxOAPG)  # focus at exit of DM telescope system

    # # Coronagraph
    SubaruPupil(wfo)  # focal plane mask
    # if kwargs['verbose'] and kwargs['ix']==0:
    #     fig, subplot = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    #     ax1, ax2 = subplot.flatten()
    #     fig.suptitle('SCExAO Model WFO after FPM', fontweight='bold', fontsize=14)
    #     # ax.imshow(dm_map, interpolation='none')
    #     ax1.imshow(np.abs(proper.prop_shift_center(wfo.wfarr))**2, interpolation='none', norm=LogNorm(vmin=1e-7,vmax=1e-2))
    #     ax1.set_title('Amplitude')
    #     ax2.imshow(np.angle(proper.prop_shift_center(wfo.wfarr)), interpolation='none',
    #                vmin=-2*np.pi, vmax=2*np.pi, cmap='hsv')  # , cmap='hsv'
    #     ax2.set_title('Phase')
    proper.prop_propagate(wfo, fl_SxOAPG)
    proper.prop_lens(wfo, fl_SxOAPG)
    proper.prop_propagate(wfo, fl_SxOAPG)  # middle of 2f system
    proper.prop_circular_aperture(wfo, lyot_size, NORM=True)  # lyot stop
    proper.prop_propagate(wfo, fl_SxOAPG)  #
    proper.prop_lens(wfo, fl_SxOAPG)  # exit lens of gaussian telescope
    proper.prop_propagate(wfo, fl_SxOAPG)  # to focus

    # MEC Pickoff reimager.
    proper.prop_propagate(wfo, mec_parax_fl)  # to another pupil
    proper.prop_lens(wfo, mec_parax_fl)  # collimating lens, pupil size should be 8 mm
    proper.prop_propagate(wfo, mec1_fl+.0142557)  # mec1_fl  .054  mec1_fl+.0101057
    # if kwargs['verbose'] and kwargs['ix']==0:
    #     current = proper.prop_get_beamradius(wfo)
    #     print(f'Beam Radius after SCExAO exit (at MEC foreoptics entrance) is {current*1e3:.3f} mm\n'
    #           f'current f# is {proper.prop_get_fratio(wfo):.2f}\n')

    # ##################################
    # MEC Optics Box
    # ###################################
    proper.prop_circular_aperture(wfo, 0.00866)  # reading off the zemax diameter
    proper.prop_lens(wfo, mec1_fl)  # MEC lens 1
    proper.prop_propagate(wfo, mec_l1_l2)  # there is a image plane at z=mec1_fl
    proper.prop_lens(wfo, mec2_fl)  # MEC lens 2 (tiny lens)
    proper.prop_propagate(wfo, mec_l2_l3)
    proper.prop_lens(wfo, mec3_fl)  # MEC lens 3
    proper.prop_propagate(wfo, mec3_fl, TO_PLANE=False)  # , TO_PLANE=True mec_l3_focus

    # #######################################
    # Focal Plane
    # #######################################
    # Check Sampling in focal plane
    # shifts wfo from Fourier Space (origin==lower left corner) to object space (origin==center)
    # wf, samp = proper.prop_end(wfo, NoAbs=True)
    wf = proper.prop_shift_center(wfo.wfarr)
    wf = extract_center(wf, new_size=np.array(kwargs['psf_size']))
    samp = proper.prop_get_sampling(wfo)
    smp_asec = proper.prop_get_sampling_arcsec(wfo)

    if kwargs['verbose'] and kwargs['ix'] == 0:
        fig, subplot = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
        fig.subplots_adjust(left=0.08, hspace=.4, wspace=0.2)

        ax1, ax2 = subplot.flatten()
        fig.suptitle('SCExAO Model Focal Plane', fontweight='bold', fontsize=14)
        tic_spacing, tic_labels, axlabel = scale_lD(wfo, newsize=kwargs['psf_size'][0])
        tic_spacing[0] = tic_spacing[0] + 1  # hack for edge effects
        tic_spacing[-1] = tic_spacing[-1] - 1  # hack for edge effects

        im = ax1.imshow(np.abs(wf)**2, interpolation='none',
                  norm=LogNorm(vmin=1e-7,vmax=1e-2))  # np.abs(proper.prop_shift_center(wfo.wfarr))**2
        ax1.set_xticks(tic_spacing)
        ax1.set_xticklabels(tic_labels)
        ax1.set_yticks(tic_spacing)
        ax1.set_yticklabels(tic_labels)
        ax1.set_ylabel(axlabel, fontsize=8)
        add_colorbar(im)

        im = ax2.imshow(np.angle(wf), interpolation='none',
                   vmin=-np.pi, vmax=np.pi, cmap='hsv')
        ax2.set_xticks(tic_spacing)
        ax2.set_xticklabels(tic_labels)
        ax2.set_yticks(tic_spacing)
        ax2.set_yticklabels(tic_labels)
        ax2.set_ylabel(axlabel, fontsize=8)
        add_colorbar(im)

    if kwargs['verbose'] and kwargs['ix']==0:
        print(f"\nFocal Plane\n"
              f"sampling at focal plane is {samp*1e6:.1f} um ~= {smp_asec * 1e3:.4f} mas\n"
              f"\tfull FOV is {samp * kwargs['psf_size'][0] * 1e3:.2f} x {samp * kwargs['psf_size'][1] * 1e3:.2f} mm ")
        # s_rad = proper.prop_get_sampling_radians(wfo)
        # print(f"sampling at focal plane is {s_rad * 1e6:.6f} urad")
        print(f'final focal ratio is {proper.prop_get_fratio(wfo)}')

        print(f"Finished simulation")

    return wf, samp



"""
# use PSF tilt for checking PSF
    # dm_map = proper.prop_zernikes(wfo, [2, 3], np.array([5, 1]))  # zernike[2,3] = x,y tilt
    # print(f'dm_map shape using zernikes is {dm_map.shape}')
    
    
 # # Create mask to eliminate resampling artifacts outside of beam
    # h, w = wfo.wfarr.shape[:2]
    # center = (int(w / 2), int(h / 2))
    # radius = np.ceil(h * beam_ratio / 2)  #
    # # Making the Circular Boolean Mask
    # Y, X = np.mgrid[:h, :w]
    # dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    # inds = dist_from_center <= radius
    # # Applying the Mask to the dm_map
    # # mask = np.zeros_like(wfo.wfarr)
    # mask = np.zeros((h,w))
    # mask[inds] = 1
    # phs = np.angle(proper.prop_shift_center(wfo.wfarr))
    # phs *= mask
"""