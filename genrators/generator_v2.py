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
    def generate_event(self):
        pass

    def run(self, num_events, n_jobs) -> list[Any]:
        
        for _ in range(num_events):
            self.generate_event()
        

        return res