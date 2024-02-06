import numpy as np

bins_file = open('/Volumes/Transcend/Documents/Study/particles/exp_data/na49/binning.bin', 'r')
bins = []
for num, line in enumerate(bins_file.readlines()):

    if num % 2 == 0:
        
        a = line.split('|')

        for bin in a:

            bin = bin.replace('(', '').replace(')', '').replace(' ', '')
            bins.append(list(map(float, bin.split(','))))
bins = np.array(bins)
bins_file.close()

mp = 0.93827                                                           #proton mass
pz = 158.0                                                             #beam momentum
root_s = (2 * mp * (mp + (mp ** 2 + pz ** 2) ** 0.5)) ** 0.5           #sqrt(s)
norm = 3.141593 * root_s / 2                                           #normalization in new coordinats


def analyze(gen_res, sigma, n_events):

    data = np.concatenate(gen_res, axis=1)

    xsect = []
    err = []

    for bin in bins:

        xF = bin[0]
        pT = bin[1]

        dxF = bin[2]
        dpT = bin[3]

        mask_for_bin = (data[1] > xF - dxF / 2) * \
            (data[1] < xF + dxF / 2) * \
            (data[2] > pT - dpT / 2) * \
            (data[2] < pT + dpT / 2)

        dn = np.sum(mask_for_bin)

        dp3 = dxF * 2 * pT * dpT

        E = data[3][mask_for_bin]

        if dn >= 2:

            xsect.append(E.mean() * (sigma / n_events) * (dn / dp3) / norm)
            err.append(xsect[-1] / (dn) ** 0.5)

        else:

            xsect.append(0)
            err.append(0)


    xsect = np.where(np.isnan(xsect), 0, xsect)
    err = np.where(np.isnan(err), 0, err)

    return xsect, err



