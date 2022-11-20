from ..methods.clip import CLIP
from ..methods.vicreg import VICREG
from ..methods.simclr import SIMCLR
from ..methods.slip_vicreg import SLIP_VICREG
from ..methods.slip_simclr import SLIP_SIMCLR
#from ..methods.Perceiver import Perceiver


METHODS = {
    # methods
    "CLIP": CLIP,
    "SLIP_VICREG": SLIP_VICREG,
    "SLIP_SIMCLR": SLIP_SIMCLR,
 #   "Perceiver": Perceiver,
    "VICREG": VICREG,
    "SIMCLR": SIMCLR,
}
__all__ = [
    "CLIP",
    "SLIP_VICREG",
    "SLIP_SIMCLR",
 #   "Perceiver",
    "VICREG",
    "SIMCLR",
]

