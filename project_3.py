import numpy as np
import matplotlib.pyplot as plt
import statistics

# https://radzion.com/blog/probability/method
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
# https://en.wikipedia.org/wiki/Method_of_moments_(statistics)
#We found this that may be interesting

s1=np.random.exponential((1/10),1000)
s2=np.random.exponential((1/50),1000)

s= s1
s= np.append.s2

mean = sum(s)/len(s)
variance = statistics.variance(s)

lambda_mean = 1/(sum(s)/len(s))
print(lambda_mean)
# lambda_variance = 


plt.hist(s1,label=r"$\lambda=10$",alpha=0.5)
plt.hist(s2,label=r"$\lambda=50$",alpha=0.5)
plt.hist(s1+s2,label="Sum of both",alpha=0.5, color="purple")
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("counts")
plt.show()


