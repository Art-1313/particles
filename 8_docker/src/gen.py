import sys
sys.path.insert(0, '/pythia8310/lib')

import pythia8
import abc_gen as g


class Gen(g.AbstractGenerator):

    def generate_event(self) -> list:

        ID = []
        PT = []
        XF = []
        EE = []

        mp = 0.93827
        pz = 158.0
    
        root_s = (2 * mp * (mp + (mp ** 2 + pz ** 2) ** 0.5)) ** 0.5

        p1 = pythia8.Vec4(0.0, 0.0, 158.0, (158.0 ** 2 + mp ** 2) ** 0.5)
        p2 = pythia8.Vec4(0.0, 0.0, 0.0, mp)
        
        boost = pythia8.RotBstMatrix()
        boost.bstback(p1 + p2)

        for _ in range(1):
            
            if not self.pythia.next(): continue
                    
            entries = self.pythia.event.size()

            for j in range(entries):

                particle = self.pythia.event[j]

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

        return list([ID, XF, PT, EE])    


def generate(instructions: dict, num_events: int, random_state: int=None) -> list:

    gen = Gen(instructions=instructions, random_state=random_state)
    return gen.run(num_events=num_events)