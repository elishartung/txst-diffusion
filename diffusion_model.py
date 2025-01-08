#!/usr/bin/env python
# coding: utf-8

# # A 1D Diffusion model

# Here we develop a one-dimensional model of dissuion. 
# It assumes constatn diffusivity. 
# It uses a regular grid.
# It has fixed boundary conditions.

# The diffusion equation:
# 
# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$
# 
# The discretized version of the diffusion equation that we'll solve with our model:
# 
# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$
# 
# This is the explicit FTCS scheme as described in Slingerland and Kump (2011). (Or see Wikipedia.)
# 

# WE will usd two libraries. Numpy (for arrays) and Matplotlib (for plotting) that aren't a part of the base Python distrubtion.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# Set two fixed modl parameters, the diffusivity and the size of the model domain

# In[ ]:


D = 100 
Lx = 300


# Next, set up the model grid using numpy array

# In[ ]:


dx=0.5
x= np.arange(start=0,stop=Lx,step=dx)
nx=len(x)


# In[ ]:


whos


# Set the initial concentration profile for the model. The concentration `C` is a step function with a high value on the left a low vaalue on the right.. and the step at the center of the domain

# In[ ]:


C = np.zeros_like(x)
C_left = 500
C_right= 0 
C[x<= Lx//2] = C_left
C[x > Lx//2] = C_right


# PLot the initial profile

# In[ ]:


plt.figure()
plt.plot(x,C,"r")
plt.xlabel("X")
plt.ylabel("C")
plt.title("Initial concentration profile")


# Set the time of the model and the number of the steps. Calculate a stable time step for the model using a stability criteria. 

# In[ ]:


time = 0
nt = 5000
dt =  0.5 * dx**2 / D


# In[ ]:


dt


# loop over the time steeps of the model, solving the diffusion equation  using the  FTCS explicit scheme described above. the boundary conditions are fixed so reset them at each time step.

# In[ ]:


for t in range(0, nt):
    C+= D * dt / dx**2 * (np.roll(C, -1) - 2*C + np.roll(C,1))
    C[0] = C_left
    C[-1] = C_right
                      


# In[ ]:


plt.figure()
plt.plot(x,C,"b")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Final Concentration Profile")


# In[ ]:


z =np.arange(5)


# In[ ]:


z


# In[ ]:


np.roll(z, -1)


# In[ ]:


np.roll(z,1)


# In[ ]:




