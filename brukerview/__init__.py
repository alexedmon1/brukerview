"""brukerview: look at Bruker ParaVision images without converting them first."""

from .reader import BrukerImage, FrameGroup
from .study import find_images, resolve_image
from .jcampdx import read_jcampdx
from .nifti import NiftiImage, find_niftis

__all__ = ['BrukerImage', 'FrameGroup', 'NiftiImage', 'find_images', 'find_niftis',
           'resolve_image', 'read_jcampdx']
__version__ = '0.1.0'
