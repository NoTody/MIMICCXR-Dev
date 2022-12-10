from ..methods.base import BASE
from ..methods.clip import CLIP
from ..methods.vicreg import VICREG
from ..methods.simclr import SIMCLR
from ..methods.mocov2 import MOCOV2
from ..methods.slip_vicreg import SLIP_VICREG
from ..methods.slip_simclr import SLIP_SIMCLR
from ..methods.slip_mocov2 import SLIP_MOCOV2
#from ..methods.Perceiver import Perceiver


METHODS = {
    # methods
    "BASE": BASE,
    "CLIP": CLIP,
    "SLIP_VICREG": SLIP_VICREG,
    "SLIP_SIMCLR": SLIP_SIMCLR,
    "SLIP_MOCOV2": SLIP_MOCOV2,
 #   "Perceiver": Perceiver,
    "VICREG": VICREG,
    "SIMCLR": SIMCLR,
    "MOCOV2": MOCOV2
}
__all__ = [
    "BASE",
    "CLIP",
    "SLIP_VICREG",
    "SLIP_SIMCLR",
    "SLIP_MOCOV2",
 #   "Perceiver",
    "VICREG",
    "SIMCLR",
    "MOCOV2",
]

