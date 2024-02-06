import sys

from typing import Any
sys.path.insert(0, '/opt/homebrew/Cellar/pythia/8.309/lib')
import pythia8

from multiprocessing import Pool    
from abc import ABC, abstractmethod
import numpy as np

class AbstractGenerator(ABC):

    def __init__(self, instructions, random_state=None) -> None:
        super().__init__()
        self.instructions = instructions
        self.random_state = random_state
    
    @abstractmethod
    def generate_event(self, N):
        return __pythia_g

    @staticmethod
    def _set_pythia(instructions, random_state=None):
        global __pythia_g
        __pythia_g = pythia8.Pythia("", False)
        __pythia_g.readString('Random:setSeed  = on')
        for inst, val in instructions.items():    
            __pythia_g.readString(f'{inst} = {val}')
        if random_state is None:
            __pythia_g.readString(f'Random:seed  = {np.random.randint(0, 90000000)}')
        else:
            __pythia_g.readString(f'Random:seed  = {random_state}')
        __pythia_g.init()

    def run(self, num_events, n_jobs):
        
        N = list(range(num_events))

        pool = Pool(
            n_jobs, 
            initializer=self._set_pythia, 
            initargs=(self.instructions, self.random_state),
        )
        res = pool.map(self.generate_event, N)
        pool.close()

        return res