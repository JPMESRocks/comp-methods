import numpy as np
import matplotlib.pyplot as plt
import statistics
import scipy

# https://radzion.com/blog/probability/method
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
# https://en.wikipedia.org/wiki/Method_of_moments_(statistics)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html
# We found this that may be interesting


def k_moment(data, k=1):
    """
        This function returns the estimate of the kth moment of the data set.
    """
    n=len(data)
    sum=np.sum(data**k)
    return (1/n)*sum

#Generating the two random datasets of exponential distributions (lambda=10,lambda=50)
s1 = np.random.exponential((1/10),1000)
s2 = np.random.exponential((1/50),1000)

#mixing data sets
s = np.append(s1, s2)

#We are trying to estimate two parameters so we only need 2 moments.
print(s)
(m1,m2)= (k_moment(s,1),k_moment(s,2))
print(m1)
print(m2)
print((m2/2 - m1**2))
# Using the formulas we derived:
lambda1=(m1+np.sqrt(0.5*m2-m1**2))**(-1)
lambda2=(m1-np.sqrt(0.5*m2-m1**2))**(-1)

print(rf"From applying the method of moments on this randomly generated exponential mixed dataset we obtain: lambda_1={lambda1} lambda_2={lambda2}")
bins = [5,10,50,100]
fig,ax=plt.subplots(1, 4, figsize=(15,4), constrained_layout=True)
for k,i in enumerate(bins):
    n = ax[k].hist(s, i, [0,1],color='purple')
fig.suptitle("Histograms of mixed data")
# plt.show()

# Method of least squares, a=lambda1, b=lambda2
f = lambda x,a, b: 0.5*(a*np.exp(-a*x) + b*np.exp(-b*x))

def chi_s(f, a, b, x, y, yerr):
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

def least_squares(f, x, y, yerr, interv, interv2, acc):
    """

    Calculates the optimal parameters a & b for funtion f to fit on certain dataset using the chi squared test.
    (optional) Draws a contour plot of the possible parameters and their chi squared values. Off by default.
    
    WARNING: Drawing a contour plot DOES change the axes that matplotlib is currently working on.
    
    Arguments:
            f       (function)                  The function of which you want to determine its parameters
            x       (array-like)                Dataset of x values
            y       (array-like)                Dataset of corresponding y values
            yerr    (array-like)                Dataset containing the corresponding error in those y values
            interv  (array-like of size 2)      a will be determined between this interval
            interv2 (array-like of size 2)      b will be determined between this interval
            acc     (int)                       Accuracy of the test. The space of the interval will be divided into this many parts to perform the test on.
    Returns:
            a, b    (floats)                    The parameters of f for optimal fit through the dataset

    """
    # We use a meshgrid because it's way way faster and simpler than a nested for-loop. 
    # TECHNICALLY this still loops over the interval and stores it in a square array, satisfying the exercise.
    av,bv = np.meshgrid(np.linspace(interv[0], interv[1], acc),np.linspace(interv2[0], interv2[1], acc), indexing='xy')
    chisquared = chi_s(f, av, bv, x, y, yerr)
    # Get the index containing the lowest chisquared value. We have to unravel because np.argmin flattens the whole 2d array.
    index = np.unravel_index(np.argmin(chisquared),chisquared.shape)
    a = av[index]
    b = bv[index]
    return a,b

x = np.linspace(0,1, len(s))
a,b = least_squares(f, x, n, np.zeros(len(s)), [8,60], [8, 60], len(s))
# print(a,b)
print(f"Least squares: {a,b}")

for i in bins:
    plt.hist(s, bins=i, density=True)
    plt.plot(x, f(x, a, b))
    plt.show()

plt.legend()
plt.xlabel("time (s)")
plt.ylabel("counts")