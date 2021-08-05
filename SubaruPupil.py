"""
SubaruPupil.py
Kristina Davis

"""

import proper

def SubaruPupil(wf):
    """
    adds Subaru pupil mask to the optical train

    :param wf: 2D proper wavefront
    :return: acts upon wfo, applies a spatial mask of s=circular secondary obscuration and possibly spider legs
    """
    # dprint('Applying Subaru Pupil')

    # M2 shadow
    proper.prop_circular_obscuration(wf, 14/46, NORM=True)
    # Legs
    proper.prop_rectangular_obscuration(wf, 1.2, 2/46, .5, -.375, ROTATION=-50, NORM=True)
    proper.prop_rectangular_obscuration(wf, 1.2, 2/46, .5, .375, ROTATION=50, NORM=True)
    proper.prop_rectangular_obscuration(wf, 1.2, 2/46, -.5, -.375, ROTATION=50, NORM=True)
    proper.prop_rectangular_obscuration(wf, 1.2, 2/46, -.5, .375, ROTATION=-50, NORM=True)
    proper.prop_rectangular_obscuration(wf, 1, 1/46, .05, .45, ROTATION=-50, NORM=True)
    # Misc Spots
    proper.prop_circular_obscuration(wf, .075, -.1, 0.6, NORM=True)
    proper.prop_circular_obscuration(wf, .075, .5, -.375, NORM=True)
