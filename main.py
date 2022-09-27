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
max_t=20
dt=0.1
ticks=int(max_t/dt)

#initializing vector lists
a=np.zeros((2,ticks)); v=np.zeros((2,ticks)); r=np.zeros((2,ticks))

#initial conditions
theta=60*(np.pi/180)
init_v=60
v[0,0]=init_v*np.cos(theta); v[1,0]=init_v*np.sin(theta)
a[1,0]=-g

#helpful math functions
sinarctan= lambda x: x/((1+x**2)**0.5)
cosarctan= lambda x: (1+x**2)**-0.5


t_space = np.linspace(0,max_t,ticks)

#calculating position of golfball numerically step-by-step
for i in range(len(t_space)-1):
    a[0,i+1]= -((v[0,i]**2+v[1,i]**2)**0.5/m)*(prop_l*sinarctan(v[1,i]/v[0,i])+prop_d*cosarctan(v[1,i]/v[0,i]))
    a[1,i+1]= ((v[0,i]**2+v[1,i]**2)**0.5/m)*(prop_l*cosarctan(v[1,i]/v[0,i])-prop_d*sinarctan(v[1,i]/v[0,i]))-g

    v[0,i+1] = v[0,i] + a[0,i+1] * dt
    v[1,i+1] = v[1,i] + a[1,i+1] * dt
    
    r[0,i+1] = r[0,i] + v[0,i+1] * dt
    r[1,i+1] = r[1,i] + v[1,i+1] * dt

    if r[1,i+1] + v[1,i+1] * dt <=0 and i>1: # Either we stop the model just before it hits the ground or it goes through the ground briefly. I chose the former.
        break

r=r[:,:i+5]; v=v[:,:i+5];a=a[:,:i+5]; t_space=t_space[:i+5]
plt.quiver(r[0,:i],r[1,:i],v[0,:i],v[1,:i],color='r',units='xy',scale=0.7,width=1,headwidth=10,headlength=30)
plt.title("Velocity vectors along trajectory")
plt.ylabel("y")
plt.xlabel("x")
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,4), constrained_layout=True)

ax1.plot(t_space, r[0]) # It plots 3x now instead of subplots cuz im lazy hf hf. Let's change this later into subplots.
ax1.set_ylabel("x")
ax1.set_xlabel("t")

ax2.plot(t_space, r[1])
ax2.set_ylabel("y")
ax2.set_xlabel("t")

ax3.plot(t_space, (a[0]**2 + a[1]**2)**0.5) # We plot the magnitude cuz idc about the individual directons.
ax3.set_ylabel(r"$|\vec{a}|$")
ax3.set_xlabel("t")

plt.show()