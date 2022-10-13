import numpy as np

def res(params, f, bin_edges, hist, yerr):
    """

    Calculates the chi squared value of certain parameters for fit function s on histogram hist.

    Arguments:
            f       (function)                  The function that we want to test the parameters on
            x       (array-like)                Dataset of x values
            y       (array-like)                Dataset of corresponding y values
            yerr    (array-like)                Dataset containing the corresponding error in those y values
    Returns:
            result  (float)                     The chi squared value of parameters a & b on function f of the dataset.

    """
    result = 0
    N = np.sum(hist)
    cbins = (bin_edges[:-1]+bin_edges[1:])/2   # calculate centres of the bins
    prediction = f(cbins, params[0], params[1]) # use those centres as x values

    for i in range(0, len(hist)):
        result += ((hist[i] - prediction[i])**2/(yerr[i])**2)
    return result