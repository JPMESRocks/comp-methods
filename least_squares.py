import numpy as np
def chi_s(f, a, b, x, y):
    """

    Calculates the chi squared value of certain parameters a and b on a dataset and function.

    Arguments:
            f       (function)                  The function that we want to test the parameters on
            a       (scalar or array-like)      First parameter to go into function f
            b       (scalar or array-like)      Second parameter to go into function f
            x       (array-like)                Dataset of x values
            y       (array-like)                Dataset of corresponding y values
            yerr    (array-like)                Dataset containing the corresponding error in those y values
    Returns:
            result  (float)                     The chi squared value of parameters a & b on function f of the dataset.

    """
    prediction = [f(x_val,a,b) for x_val in x]
    result = 0
    for i in range(0, len(y)):
        result += ((y[i] - prediction[i])**2/prediction[i])
    return result

def least_squares(f, x, y, interv, acc):
    """

    Calculates the optimal parameters a & b for funtion f to fit on certain dataset using the chi squared test.
    (optional) Draws a contour plot of the possible parameters and their chi squared values. Off by default.
    
    WARNING: Drawing a contour plot DOES change the axes that matplotlib is currently working on.
    
    Arguments:
            f       (function)                  The function of which you want to determine its parameters
            x       (array-like)                Dataset of x values
            y       (array-like)                Dataset of corresponding y values
            yerr    (array-like)                Dataset containing the corresponding error in those y values
            interv  (array-like of size 2)      a and b will be determined between this interval
            acc     (int)                       Accuracy of the test. The space of the interval will be divided into this many parts to perform the test on.
    Returns:
            a, b    (floats)                    The parameters of f for optimal fit through the dataset

    """
    # We use a meshgrid because it's way way faster and simpler than a nested for-loop. 
    # TECHNICALLY this still loops over the interval and stores it in a square array, satisfying the exercise.
    av,bv = np.meshgrid(np.linspace(interv[0], interv[1], acc),np.linspace(interv[0], interv[1], acc), indexing='xy')
    chisquared = chi_s(f, av, bv, x, y)
    # Get the index containing the lowest chisquared value. We have to unravel because np.argmin flattens the whole 2d array.
    index = np.unravel_index(np.argmin(chisquared),chisquared.shape)
    a = av[index]
    b = bv[index]
    return a,b