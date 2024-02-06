import sys
sys.path.insert(0, '/opt/homebrew/Cellar/pythia/8.309/lib')
import pythia8

import numpy as np

from abc import ABC, abstractmethod

class AbstartcPythiaGenerator(ABC):

    def __init__(self, instructions: dict, n_jobs=0) -> None:
        super().__init__()

        self.pythia = pythia8.PythiaParallel("", False)
        
        self.pythia.readString(f'Parallelism:numThreads  = {n_jobs}')
        self.pythia.readString(f'Random:setSeed  = on')

        for inst, val in instructions.items():
        
            self.pythia.readString(f'{inst} = {val}')

    @abstractmethod
    def analize_event(self, pythia_ptr):
        pass

    def generate(self, n_events: int, random_state: int=None):
        if random_state is None:
            self.pythia.readString(f'Random:seed  = {np.random.randint(0, 90000000)}')
        else:
            self.pythia.readString(f'Random:seed  = {random_state}')
        self.pythia.init()
        self.pythia.run(n_events, self.analize_event)
        
