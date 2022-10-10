import numpy as np
import matplotlib.pyplot as plt
import statistics
import scipy

# https://radzion.com/blog/probability/method
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
# https://en.wikipedia.org/wiki/Method_of_moments_(statistics)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html
# We found this that may be interesting


s1 = np.random.exponential((1/10),1000)
s2 = np.random.exponential((1/50),1000)

s = s1
np.append(s, s2)

mean = sum(s)/len(s)
variance = statistics.variance(s)

lambda_mean = 1/(sum(s)/len(s))
print(lambda_mean)

# plt.hist(s1,label=r"$\lambda=10$",alpha=0.5)
# plt.hist(s2,label=r"$\lambda=50$",alpha=0.5)
# plt.hist(s1+s2,label="Sum of both",alpha=0.5, color="purple")

for i in [5,10,50,100]:
    plt.hist(s, bins=i)
    plt.show()

plt.legend()
plt.xlabel("time (s)")
plt.ylabel("counts")