import numpy as np
import matplotlib.pyplot as plt
print('I am a golf ball and I want to fly.')


# Necessary constants (all in SI units)
m=0.05
r=0.02
g=9.81
rho=1.2
A= np.pi*r**2

def golfball(initialangle: float =60, drag: bool = True, lift: bool = True):
    """

    Numerically calculates the trajectory of a golfball being projected at some angle, taking into account
    drag and lift. Plots the x-y trajectory and graphs the x and y components of acceleration throughout its flight.

    Arguments:
                initialangle    (float)         The angle in degrees at which the ball is "thrown"
                drag            (boolean)       Whether the drag force is calculated and taken into account
                lift            (boolean)       Whether the lift force is calculated and taken into account
    Returns:
            None

    """
    # setting drag/lift
    c_d=0.3 if drag else 0
    c_l=0.3 if lift else 0

    prop_l=0.5*rho*A*c_l
    prop_d=0.5*rho*A*c_d

    # time-steps for the diff. eq.
    max_t=20
    dt=0.001
    ticks=int(max_t/dt)

    # initializing vector lists
    a=np.zeros((2,ticks)); v=np.zeros((2,ticks)); r=np.zeros((2,ticks))
    t_space = np.linspace(0,max_t,ticks)

    # initial conditions. (change init angle here)
    conv_angle=(np.pi/180)*initialangle
    init_v=60
    v[0,0]=init_v*np.cos(conv_angle); v[1,0]=init_v*np.sin(conv_angle)
    a[1,0]=-g


    # helpful math functions
    sinarctan= lambda x: x/((1+x**2)**0.5)
    cosarctan= lambda x: (1+x**2)**-0.5


    # calculating position of golfball numerically step-by-step
    for i in range(len(t_space)-1):
        a[0,i+1]= -((v[0,i]**2+v[1,i]**2)**0.5/m)*(prop_l*sinarctan(v[1,i]/v[0,i])+prop_d*cosarctan(v[1,i]/v[0,i]))
        a[1,i+1]= ((v[0,i]**2+v[1,i]**2)**0.5/m)*(prop_l*cosarctan(v[1,i]/v[0,i])-prop_d*sinarctan(v[1,i]/v[0,i]))-g

        v[0,i+1] = v[0,i] + a[0,i+1] * dt
        v[1,i+1] = v[1,i] + a[1,i+1] * dt
        
        r[0,i+1] = r[0,i] + v[0,i+1] * dt
        r[1,i+1] = r[1,i] + v[1,i+1] * dt

        if r[1,i+1] + v[1,i+1] * dt <=0 and i>1: # Stopping the calculations right before the ball hits the ground
            r[1,i+1]=0
            r[0,i+1]= r[0,i] + v[0,i+1] * dt
            break
    # splicing the empty elements out 
    r=r[:,:i+2]; v=v[:,:i+1];a=a[:,:i+1]; t_space=t_space[:i+1] 
    print(a)
    # Plotting trajectory and acceleration components
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,4), constrained_layout=True)

    ax1.plot(r[0],r[1])
    ax1.axis('scaled')
    ax1.set_ylabel("y (m)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylim(-10,200)
    ax1.set_xlim(-10,350)

    ax2.plot(t_space, a[0])
    ax2.set_ylabel(r"$a_x\;(ms^{-2})$")
    ax2.set_xlabel("t (s)")

    ax3.plot(t_space, a[1]) 
    ax3.set_ylabel(r"$a_y\;(ms^{-2})$")
    ax3.set_xlabel("t (s)")

    fig.suptitle(f"initial angle: {initialangle}, drag: {drag}, lift: {lift}")

    print(f"Flight time: {round(t_space[i],2)}s, Distance travelled: {round(r[0,i],2)}m")
    plt.show()


# In each component of acceleration there is a jump in the first time step because of 
# the initial condition being different(a_x=0, a_y=-g). This is just an artifact of our numerical method. 
golfball(initialangle=60,drag=True,lift=True)

# Interestingly, if we only disable drag. The non-linearity of the y-acceleration becomes more visible
golfball(initialangle=60, drag=False, lift=True)

# For the ball being shot straight up (initialangle=90), the trajectory is correct and makes sense. However, the acceleration graphs behave very weirdly, forming a solid triangle.
# When looking at the values of the acceleration themselves, the drag+lift seem to oscillate between positive and negative values in each acceleration component.

# I believe this occurs because these forces depend on the ball's velocity, who's x-component is initially zero. In the first step, lift pushes the ball into a negative x-velocity,
# As soon as this occurs, that negative x-velocity causes a stronger drag force pushing it in the +x direction bringing it back to around 90 degrees. This occurs over and over
# until eventually the lift dominates at 6s and the ball falls around x=-1.7m.
golfball(initialangle=90,drag=True,lift=True)

# Same thing, different angle. Everything lookin good.
golfball(initialangle=20,drag=True,lift=True)

# shooting the golfball parallel to the ground at the ground provides the expected result considering our loop's stop statement. The loop iterates once
# until it realizes the ball is on the ground and stops.
golfball(initialangle=0,drag=True,lift=True)
