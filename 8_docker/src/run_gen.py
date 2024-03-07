import gen
import os

import numpy as np

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
    'TimeShower:phiPolAsym': 'on'
}


if __name__ == '__main__':

    N = int(os.environ['PYTHIA_N'])
    PARAMS = os.environ['PYTHIA_PARAMS']
    #OUTPUT_DIR = os.environ['PYTHIA_OUTPUT_DIR']

    inst.update(
        dict(
            map(str.split, PARAMS.split(', '))
        )
    )

    gen_res = gen.generate(inst, N)

    file = open(f'/output/output.txt', 'w')
    file.writelines('\n'.join(list(map(' '.join, np.array(np.concatenate(gen_res, axis=1).T).astype(str)))))
    file.close()