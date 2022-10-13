import numpy as np
import matplotlib.pyplot as plt
import statistics
import scipy

from least_squares import least_squares # the other file named "least_squares.py" that I shoved the function in.

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
s1 = np.random.exponential((1/10),200)
s2 = np.random.exponential((1/50),200)
#mixing data sets
s = np.append(s1, s2)

#We are trying to estimate two parameters so we only need 2 moments.
# print(s)
(m1,m2)= (k_moment(s,1),k_moment(s,2))
# print(m1)
# print(m2)
# print((m2/2 - m1**2))
# Using the formulas we derived:
lambda1=(m1+np.sqrt(0.5*m2-m1**2))**(-1)
lambda2=(m1-np.sqrt(0.5*m2-m1**2))**(-1)

bins = [5,10,50,100] # Amount of bins used for the histogram. Given in the exercise.

# Our fit function that we use everywhere; a=lambda1.
f = lambda x,a, b: 0.5*(a*np.exp(-a*x) + b*np.exp(-b*x))

def compare_fit(x, fit_y, method: str=""):
    # plots your fit function into the histograms
    # fit_y is your array of y values.
    # The "method" parameter is just to put it in the title of the plot
    # Returns n, the values of the histogram bins.

    fig,ax=plt.subplots(1, 4, figsize=(15,4), constrained_layout=True)
    for k,i in enumerate(bins):
        n = ax[k].hist(s, i,color='purple', density=True)
        ax[k].plot(x, fit_y, color='orange') # Plot the fit function
    fig.suptitle(f"Histograms of mixed data - {method}")
    plt.show()
    return n

print(rf"From applying the method of moments on this randomly generated exponential mixed dataset we obtain: lambda_1={lambda1} lambda_2={lambda2}")

x= np.linspace(0,1,100)

# n= values of the histogram which we need to have for the least squares method.
# To fit through the histogram you wanna have the values of the histogram :D
# n[0] is the y values and n[1] is the x values. Source: trust me bro
n = compare_fit(x, f(x, lambda1, lambda2), "Method of moments")  

x = np.linspace(0,1, len(n[0])) # the length of the last n[0] so that will be the last value in the "bins" array. This length actually matters now.
a,b = least_squares(f, x, n[0], [1,101], 1000) # Since the interval for both a and b starts at 1, we are using (1,1) as starting parameters

print(f"Least squares gives lambda1={a} & lambda2={b}")

compare_fit(x, f(x,a,b), "Least squares")