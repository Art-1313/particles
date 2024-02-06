import generator_v1 as g

import sys
sys.path.insert(0, '/opt/homebrew/Cellar/pythia/8.309/lib')
import pythia8

instructions = {
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

class Gen(g.AbstractGenerator):

    def generate_event(self, N):

        pythia = super().generate_event(N)
        ID = []
        PT = []
        XF = []
        EE = []

        mp = 0.93827                                                           #proton mass
        pz = 158.0   
    
        root_s = (2 * mp * (mp + (mp ** 2 + pz ** 2) ** 0.5)) ** 0.5           #sqrt(s)

        p1 = pythia8.Vec4(0.0, 0.0, 158.0, (158.0 ** 2 + mp ** 2) ** 0.5)
        p2 = pythia8.Vec4(0.0, 0.0, 0.0, mp)
        
        boost = pythia8.RotBstMatrix()
        boost.bstback(p1 + p2)
            
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


if __name__ == '__main__':

    gen = Gen(instructions=instructions)
    print(gen.run(num_events=int(1e0), n_jobs=8))