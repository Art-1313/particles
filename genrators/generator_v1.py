import sys

from typing import Any
sys.path.insert(0, '/opt/homebrew/Cellar/pythia/8.309/lib')
import pythia8

from multiprocessing import Pool    
from abc import ABC, abstractmethod

class AbstractGenerator(ABC):

    def __init__(self, instructions) -> None:
        super().__init__()
        self.instructions = instructions
    
    @abstractmethod
    def generate_event(self, N):
        return __pythia_g

    @staticmethod
    def _set_pythia(instructions):
        global __pythia_g
        __pythia_g = pythia8.Pythia("", False)
        for inst, val in instructions.items():    
            __pythia_g.readString(f'{inst} = {val}')
        __pythia_g.init()

    def run(self, num_events, n_jobs) -> list[Any]:
        
        N = list(range(num_events))

        pool = Pool(
            n_jobs, 
            initializer=self._set_pythia, 
            initargs=(self.instructions,),
        )
        res = pool.map(self.generate_event, N)
        pool.close()

        return res