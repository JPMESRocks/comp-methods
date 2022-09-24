from imp import init_builtin
import numpy as np
import matplotlib.pyplot as plt
print('I am a golf ball and I want to fly.')


#Necessary constants
m=0.05
r=0.02
c_l=0.3
c_d=0.3
g=9.81
rho=1.225
A= np.pi*r**2
prop_l=0.5*rho*A*c_l
prop_d=0.5*rho*A*c_d

#diff eq. parameters
max_t=100000
dt=0.01
ticks=int(max_t/dt)

#initializing vector lists
a=np.zeros((2,ticks)); v=np.zeros((2,ticks)); r=np.zeros((2,ticks))

#initial conditions
theta=60*(np.pi/180)
init_v=60
v[0,0]=init_v*np.cos(theta); v[1,0]=init_v*np.sin(theta)

#helpful math functions
sinarctan= lambda x: x/((1+x**2)**0.5)
cosarctan= lambda x: (1+x**2)**-0.5


t_space = np.linspace(0,max_t,max_t+1)

#calculating position of golfball numerically step-by-step
for i in range(len(t_space)-1):
    a[0,i+1]= -((v[0,i]**2+v[1,i]**2)**0.5/m)*(prop_l*sinarctan(v[1,i]/v[0,i])+prop_d*cosarctan(v[1,i]/v[0,i]))
    a[1,i+1]= ((v[0,i]**2+v[1,i]**2)**0.5/m)*(prop_l*cosarctan(v[1,i]/v[0,i])-prop_d*sinarctan(v[1,i]/v[0,i]))-g

    v[0,i+1] = v[0,i] + a[0,i+1] * dt
    v[1,i+1] = v[1,i] + a[1,i+1] * dt
    
    r[0,i+1] = r[0,i] + v[0,i+1] * dt
    r[1,i+1] = r[1,i] + v[1,i+1] * dt
    if r[1,i+1]<=0 and i>1:
        break
print(i)
print(v[0,i])