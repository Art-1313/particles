import sys
sys.path.insert(0, '/opt/homebrew/Cellar/pythia/8.309/lib')
import pythia8


from tqdm import tqdm
import pandas as pd
import numpy as np


mp = 0.93827                                                           #proton mass
pz = 158.0                                                             #beam momentum
root_s = (2 * mp * (mp + (mp ** 2 + pz ** 2) ** 0.5)) ** 0.5           #sqrt(s)
norm = 3.141593 * root_s / 2                                           #normalization in new coordinats
sig_trig = 28.23                                                       #trigger cross section


def xsect_calc(df: pd.DataFrame, sig_total: float, n_events: int, bins: np.ndarray):

    xsect = (((df.groupby(['id', 'bin']).mean()['E']) * \
        (df.groupby(['id', 'bin']).count()['E']))).to_frame(name='xsect').reset_index()
    
    err = (((df.groupby(['id', 'bin']).std()['E']) * \
        (df.groupby(['id', 'bin']).count()['E']))).to_frame(name='err').reset_index()

    xsect['xsect'] = (sig_total / n_events / norm) * xsect['xsect'] / \
        (2 * bins[xsect['bin']][:, 2] * bins[xsect['bin']][:, 1] * bins[xsect['bin']][:, 3])
    
    err['err'] = (sig_total / n_events / norm) * err['err'] / \
        (2 * bins[err['bin']][:, 2] * bins[err['bin']][:, 1] * bins[err['bin']][:, 3])

    return xsect, err


def chi_square(y_hat, y, y_hat_err, y_err):

    return (y_hat - y) ** 2 / (y_hat_err ** 2 + y_err ** 2)


def calc_metric(func, xsect, err, ref, bins, id):

    res = []

    for num, bin in enumerate(bins):

        mask = (ref['xF'] == bin[0]) * (ref['pT'] == bin[1])

        if sum(mask) * sum(xsect[xsect['id'] == id]['bin'] == num) != 0:

            y_hat = xsect[(xsect['bin'] == num) * (xsect['id'] == id)]['xsect'].values[0]
            y=ref['xsect'][mask].values[0]

            y_hat_err = err[(err['bin'] == num) * (err['id'] == id)]['err'].values[0]
            y_err=ref['error'][mask].values[0]

            res.append(func(y_hat=y_hat, y=y,
                            y_hat_err=y_hat_err, y_err=y_err))

    return np.where(np.isnan(res), 0, res)


def get_score(data: pd.DataFrame, sigma: float, n_events: int, bins: np.ndarray, refs: dict):

    xsect, err = xsect_calc(data, sigma, n_events, bins)

    res = []

    for id, ref in refs.items():

        res.append(np.mean(calc_metric(chi_square, xsect, err, ref, bins, id)))

    return np.sum(res)


def check_bin(xF: float, pT: float, bins: np.ndarray):

    bin_num = np.argmin((bins[:, 0] - xF) ** 2 + (bins[:, 1] - pT) ** 2)

    bin_min = bins[bin_num]
    
    if ((bin_min[0] - bin_min[2] / 2 < xF < bin_min[0] + bin_min[2] / 2) * \
        (bin_min[1] - bin_min[3] / 2 < pT < bin_min[1] + bin_min[3] / 2)):
        return bin_num
    
    else: return -1


def generate(n_events: int, instructions: dict, bins: np.ndarray):

    pythia = pythia8.Pythia("", False)

    for inst, val in instructions.items():
    
        pythia.readString(f'{inst} = {val}')

    pythia.init()

    Id = []
    PT = []
    XF = []
    EE = []
    BS = []

    boost = pythia8.Vec4()
    boost.pz(mp * ((1 + (pz / mp) ** 2) ** 0.5 - 1) ** 0.5)
    boost.e((boost.pz() ** 2 + mp ** 2) ** 0.5)
    
    for _ in tqdm(range(n_events)):

        if not pythia.next(): continue
        
        entries = pythia.event.size()

        for j in range(entries):

            particle = pythia.event[j]

            id = particle.id()
            
            if abs(int(id)) == 211 or abs(int(id)) == 321:

                P_mu = particle.p()

                P_mu.bstback(boost)

                pT = P_mu.pT()
                xF = 2 * P_mu.pz() / root_s
                E = P_mu.e()

                bin = check_bin(xF, pT, bins)

                if bin != -1:

                    Id.append(id)
                    PT.append(pT)
                    XF.append(xF)
                    EE.append(E)
                    BS.append(bin)

    data = pd.DataFrame({'id': Id, 'pT': PT, 'xF': XF, 'E': EE, 'bin': BS})
    sigma = pythia.getSigmaTotal()

    return data, sigma


def save(data: pd.DataFrame, sigma: float, dir=None):

    if not dir:

        data.to_csv(f'{dir}/data.csv', sep=';', index=False)
        
        file = open(f'{dir}/sigma.txt', 'w')
        file.write(sigma)
        file.close()
