from ..methods.clip import CLIP
from ..methods.slip import SLIP
#from ..methods.Perceiver import Perceiver


METHODS = {
    # methods
    "CLIP": CLIP,
    "SLIP": SLIP,
 #   "Perceiver": Perceiver,
}
__all__ = [
    "CLIP",
    "SLIP",
 #   "Perceiver",
]

