from gen_mp import AbstartcPythiaGenerator

import sys
sys.path.insert(0, '/opt/homebrew/Cellar/pythia/8.309/lib')
import pythia8

mp = 0.93827                                                           #proton mass
pz = 158.0                                                             #beam momentum
root_s = (2 * mp * (mp + (mp ** 2 + pz ** 2) ** 0.5)) ** 0.5           #sqrt(s)
norm = 3.141593 * root_s / 2                                           #normalization in new coordinats

p1 = pythia8.Vec4(0.0, 0.0, 158.0, (158.0 ** 2 + mp ** 2) ** 0.5)
p2 = pythia8.Vec4(0.0, 0.0, 0.0, mp)
boost = pythia8.RotBstMatrix()
boost.bstback(p1 + p2)

class Gen(AbstartcPythiaGenerator):

    def __init__(self, instructions: dict, n_jobs=0) -> None:
        super().__init__(instructions, n_jobs)

        #self.ID = []
        #self.PT = []
        #self.XF = []
        #self.EE = []

    def analize_event(self, pythia_ptr):

        entries = pythia_ptr.event.size()

        for j in range(entries):

            particle = pythia_ptr.event[j]

            id = particle.id()
            
            if abs(int(id)) == 211 or abs(int(id)) == 321:

                P_mu = particle.p()
                P_mu.rotbst(boost)

                pT = P_mu.pT()
                xF = 2 * P_mu.pz() #/ root_s
                E = P_mu.e()

                #self.ID.append(id)
                #self.PT.append(pT)
                #self.XF.append(xF)
                #self.EE.append(E)

    def extract_results(self):

        pass#return self.ID, self.PT, self.XF, self.EE
    
inst = {
    'Print:quiet': 'on',

    'Beams:frameType': '2',

    'Beams:idA': '2212',
    'Beams:idB': '2212',

    'Beams:eA': '158',
    'Beams:eB': '0',

    'SoftQCD:all': 'on',

    'Tune:pp': '1',

    'SigmaDiffractive:dampen': 'on',
    'SpaceShower:phiIntAsym': 'on',
    'SpaceShower:phiPolAsym': 'on',
    'SpaceShower:rapidityOrder': 'on',
    'SpaceShower:rapidityOrderMPI': 'on',
    'SpaceShower:samePTasMPI': 'off',
    'TimeShower:dampenBeamRecoil': 'on',
    'TimeShower:phiPolAsym': 'on',

    'MultipartonInteractions:ecmRef': '1800',
    }



from time import time
#import matplotlib.pyplot as plt

iters = 10
N = int(1e6)

file = open('mp.txt', 'w')

for i in range(1, 9):

    gen = Gen(inst, n_jobs=i)

    start = time()
    for _ in range(iters):
        gen.generate(N, 10)
    end = time()
    
    file.write(f'1 {(end - start) / iters}')
    print(f'Estimation for {i} threads completed. Time spent on one iterration is {(end - start) / iters}')

file.close()
