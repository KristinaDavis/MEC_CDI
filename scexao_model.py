"""
scexao_model.py
Kristina Davis


"""


import numpy as np
from inspect import getframeinfo, stack
import proper

from SubaruPupil import SubaruPupil
from errormap import errormap

#################################################################
# SCExAO Optial Properties
# ################################################################
# SCExAO
beam_ratio = 0.36

entrance_d = 0.018  # size of beam on the DM is 18 mm^2

# SCExAO Optics
fl_SxOAPG = 0.255  # m focal length of Genera SCExAO lens (OAP1,3,4,5)
fl_SxOAP2 = 0.519  # m focal length of SCExAO OAP 2
d_SxOAPG = 0.051  # diameter of SCExAO OAP's are 2 inches=0.051 m
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
mec1_fl = 0.1  # [m] optic 1 focal length = 100 mm
mec2_fl = 0.009 # [m] optic 2 focal length = 9 mm
mec3_fl = .3  # [m] optic 3 focal length = 300 mm


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
    if kwargs['verbose']:
        check1 = proper.prop_get_sampling(wfo)
        print(f"\n\tDM Pupil Plane\n"
              f"sampling at aperture is {check1 * 1e3:.4f} mm\n"
              f"Total Sim is {check1 * 1e3 * grid_size:.2f}x{check1 * 1e3 * grid_size:.2f} mm\n"
              f"Size of beam is {check1 * 1e3 * grid_size * beam_ratio:.4f} mm^2\n")

    ########################################
    # Import/Apply Actual DM Map
    # #######################################
    dm_map = kwargs['map']
    errormap(wfo, dm_map, SAMPLING=dm_pitch, MIRROR_SURFACE=True, MICRONS=True)

    # ------------------------------------------------
    proper.prop_propagate(wfo, fl_SxOAPG)  # from tweeter-DM to OAP2

    # SCExAO Reimaging 2
    proper.prop_propagate(wfo, fl_SxOAPG)
    proper.prop_lens(wfo, fl_SxOAP2)
    proper.prop_propagate(wfo, fl_SxOAP2)  # exits the DM telescope system

    # # Coronagraph
    # wfo.loop_collection(cg.coronagraph, occulter_mode=cg_type, plane_name='coronagraph')
    SubaruPupil(wfo)  # focal plane mask
    proper.prop_propagate(wfo, fl_SxOAPG)
    proper.prop_lens(wfo, fl_SxOAPG)
    proper.prop_propagate(wfo, fl_SxOAPG)  # middle of 2f system
    proper.prop_circular_aperture(wfo, lyot_size, NORM=True)  # lyot stop
    proper.prop_propagate(wfo, fl_SxOAPG)  #
    proper.prop_lens(wfo, fl_SxOAPG)  # exit lens of gaussian telescope
    proper.prop_propagate(wfo, fl_SxOAPG)  # to final focal plane

    ########################################
    # Focal Plane
    # #######################################
    # Check Sampling in focal plane
    # shifts wfo from Fourier Space (origin==lower left corner) to object space (origin==center)
    # proper.prop_shift_center(wfo.wfarr)
    wf, samp = proper.prop_end(wfo, NoAbs=True)
    # cpx_planes, sampling = wfo.focal_plane()
    smpling = proper.prop_get_sampling_arcsec(wfo)

    if kwargs['verbose']:
        print(f"\n\tFocal Plane\n"
              f"sampling at focal plane is {smpling * 1e3:.4f} mas\n"
              f"\tfull FOV is {smpling * grid_size * 1e3:.2f} mas")
        s_rad = proper.prop_get_sampling_radians(wfo)
        print(f"sampling at focal plane is {s_rad * 1e6:.6f} urad")

        print(f"Finished simulation")

    return wf, samp



