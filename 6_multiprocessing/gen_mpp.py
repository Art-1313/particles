import sys
sys.path.insert(0, '/opt/homebrew/Cellar/pythia/8.309/lib')
import pythia8

import numpy as np


mp = 0.93827                                                           #proton mass
pz = 158.0                                                             #beam momentum
root_s = (2 * mp * (mp + (mp ** 2 + pz ** 2) ** 0.5)) ** 0.5           #sqrt(s)
norm = 3.141593 * root_s / 2                                           #normalization in new coordinats

from abc import ABC, abstractmethod

class AbstartcPythiaGenerator(ABC):

    def __init__(self, instructions: dict) -> None:
        super().__init__()

        self.pythia = pythia8.Pythia("", False)

        self.pythia.readString(f'Random:setSeed  = on')

        for inst, val in instructions.items():
        
            self.pythia.readString(f'{inst} = {val}')

    @abstractmethod
    def generate(self, n_events: int, random_state: int=None):
        if random_state is None:
            self.pythia.readString(f'Random:seed  = {np.random.randint(0, 90000000)}')
        else:
            self.pythia.readString(f'Random:seed  = {random_state}')
        self.pythia.init()