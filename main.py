from imp import init_builtin
import numpy as np
import matplotlib.pyplot as plt
print('I am a golf ball and I want to fly.')


#Necessary constants (all in SI units)
m=0.05
r=0.02
g=9.81
rho=1.2
A= np.pi*r**2

def golfball(theta: float =60, drag: bool = True, lift: bool = True):
    #setting drag/lift
    c_d=0.3 if drag else 0
    c_l=0.3 if lift else 0

    prop_l=0.5*rho*A*c_l
    prop_d=0.5*rho*A*c_d

    #time-steps for the diff. eq.
    max_t=20
    dt=0.001
    ticks=int(max_t/dt)

    #initializing vector lists
    a=np.zeros((2,ticks)); v=np.zeros((2,ticks)); r=np.zeros((2,ticks))
    t_space = np.linspace(0,max_t,ticks)

    #initial conditions. (change init angle here)
    conv_theta=(np.pi/180)*theta
    init_v=60
    v[0,0]=init_v*np.cos(conv_theta); v[1,0]=init_v*np.sin(conv_theta)
    a[1,0]=-g


    #helpful math functions
    sinarctan= lambda x: x/((1+x**2)**0.5)
    cosarctan= lambda x: (1+x**2)**-0.5


    #calculating position of golfball numerically step-by-step
    for i in range(len(t_space)-1):
        a[0,i+1]= -((v[0,i]**2+v[1,i]**2)**0.5/m)*(prop_l*sinarctan(v[1,i]/v[0,i])+prop_d*cosarctan(v[1,i]/v[0,i]))
        a[1,i+1]= ((v[0,i]**2+v[1,i]**2)**0.5/m)*(prop_l*cosarctan(v[1,i]/v[0,i])-prop_d*sinarctan(v[1,i]/v[0,i]))-g

        v[0,i+1] = v[0,i] + a[0,i+1] * dt
        v[1,i+1] = v[1,i] + a[1,i+1] * dt
        
        r[0,i+1] = r[0,i] + v[0,i+1] * dt
        r[1,i+1] = r[1,i] + v[1,i+1] * dt

        if r[1,i+1] + v[1,i+1] * dt <=0 and i>1: #Stopping the calculations right before the ball hits the ground
            r[1,i+1]=0
            r[0,i+1]= r[0,i] + v[0,i+1] * dt
            break
    #splicing the empty elements out 
    r=r[:,:i+2]; v=v[:,:i+1];a=a[:,:i+1]; t_space=t_space[:i+1] 

    #Plotting trajectory and acceleration components
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,4), constrained_layout=True)

    ax1.plot(r[0],r[1])
    ax1.axis('scaled')
    ax1.set_ylabel("y (m)")
    ax1.set_xlabel("x (m)")

    ax2.plot(t_space, a[0])
    ax2.set_ylabel(r"$a_x\;(ms^{-2})$")
    ax2.set_xlabel("t (s)")

    ax3.plot(t_space, a[1]) 
    ax3.set_ylabel(r"$a_y\;(ms^{-2})$")
    ax3.set_xlabel("t (s)")

    fig.suptitle(f"initial angle: {theta}, drag: {drag}, lift: {lift}")

    print(f"Flight time: {round(t_space[i],2)}s, Distance travelled: {round(r[0,i],2)}m")
    plt.show()


golfball()