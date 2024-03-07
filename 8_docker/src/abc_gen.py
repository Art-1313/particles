import sys

from typing import Any
sys.path.insert(0, '/pythia8310/lib')
import pythia8

from abc import ABC, abstractmethod
import numpy as np

class AbstractGenerator(ABC):

    def __init__(self, instructions: dict, random_state: int = None) -> None:
        super().__init__()
        self.__init_pythia(instructions, random_state)
    
    def __init_pythia(self, instructions: dict, random_state: int = None) -> None:
        self.pythia = pythia8.Pythia("", False)
        self.pythia.readString('Random:setSeed  = on')
        for key, val in instructions.items():
            self.pythia.readString(f'{key} = {val}')
        if random_state is None:
            self.pythia.readString(f'Random:seed  = {np.random.randint(1, 90000000)}')
        else:
            self.pythia.readString(f'Random:seed  = {random_state}')
        self.pythia.init()
    
    @abstractmethod
    def generate_event(self):
        pass

    def run(self, num_events) -> list[Any]:

        res = []
        
        for _ in range(num_events):
            res.append(self.generate_event())

        return res