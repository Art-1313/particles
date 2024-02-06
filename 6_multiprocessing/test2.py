import sys
sys.path.insert(0, '/opt/homebrew/Cellar/pythia/8.309/lib')
import pythia8

from multiprocessing import Pool
import pandas as pd

mp = 0.93827                                                           #proton mass
pz = 158.0                                                             #beam momentum
root_s = (2 * mp * (mp + (mp ** 2 + pz ** 2) ** 0.5)) ** 0.5           #sqrt(s)
norm = 3.141593 * root_s / 2                                           #normalization in new coordinats

def generate(n):

    p1 = pythia8.Vec4(0.0, 0.0, 158.0, (158.0 ** 2 + mp ** 2) ** 0.5)
    p2 = pythia8.Vec4(0.0, 0.0, 0.0, mp)
    boost = pythia8.RotBstMatrix()
    boost.bstback(p1 + p2)
    
    ID = []
    PT = []
    XF = []
    EE = []

    for _ in range(1):
        
        if not pythia.next(): continue
                
        entries = pythia.event.size()

        for j in range(entries):

            particle = pythia.event[j]

            id = particle.id()
            
            if (int(id) == 211) and particle.isFinal():

                P_mu = particle.p()
                P_mu.rotbst(boost)

                pT = P_mu.pT()
                xF = 2 * P_mu.pz() / root_s
                E = P_mu.e()

                ID.append(id)
                PT.append(pT)
                XF.append(xF)
                EE.append(E)

    return list([ID, PT, XF, EE])

def set_pythia(instructions):

    global pythia

    pythia = pythia8.Pythia("", False)
    for inst, val in instructions.items():    
        pythia.readString(f'{inst} = {val}')
    pythia.init()

from time import time

def run(instructions, num_events, n_jobs):

    if __name__ == 'test2':
        
        N = list(range(num_events))

        start = time()

        pool = Pool(
            n_jobs, 
            initializer=set_pythia, 
            initargs=(instructions,)
        )
        res = pool.map(generate, N)

        end = time()

        print(end - start)

        ID = []
        PT = []
        XF = []
        EE = []

        for r in res:

            for num in range(len(r[0])):

                ID.append(r[0][num])
                PT.append(r[1][num])
                XF.append(r[2][num])
                EE.append(r[3][num])

        df = pd.DataFrame({'id': ID, 'pT': PT, 'xF': XF, 'E': EE})
        return df