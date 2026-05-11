from dataclasses import replace
from typing import Optional

import numpy as np
import random as rd
from scipy.spatial.distance import cdist

from proteor1.generate.data_load import const
from proteor1.generate.data_load.crop.cropper import Cropper
from proteor1.generate.data_load.crop.boltz import BoltzCropper
from proteor1.generate.data_load.crop.antibody import AntibodyCropper
from proteor1.generate.data_load.types import Tokenized


class MixedCropper(Cropper):

    def __init__(self, add_antigen: Optional[bool] = False,
                 min_neighborhood: int = 0, max_neighborhood: int = 40,
                 probability: float = 0.5) -> None:

        self.antibody_cropper = AntibodyCropper(add_antigen, min_neighborhood, max_neighborhood)
        self.boltz_cropper = BoltzCropper(min_neighborhood, max_neighborhood)
        self.p = probability

    def crop(  # noqa: PLR0915
        self,
        data: Tokenized,
        token_mask: np.ndarray,
        token_region: np.ndarray,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        h_chain_id: Optional[int] = None,
        l_chain_id: Optional[int] = None,
    ):

        chosen = rd.choices([0, 1], weights=[self.p, 1-self.p])[0]
        if chosen == 0:
            return self.antibody_cropper.crop(
                data, token_mask, token_region, max_tokens,
                random, max_atoms, chain_id, h_chain_id, l_chain_id
            )
        else:
            return self.boltz_cropper.crop(
                data, token_mask, token_region, max_tokens,
                random, max_atoms
            )
