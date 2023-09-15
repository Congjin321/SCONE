import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the involute function
def involuter(r, t, a):
    x = r * (np.cos(t + a) + t * np.sin(t + a))
    y = r * (np.sin(t + a) - t * np.cos(t + a))
    return x, y

def intesection_curve(x0, y0, theta, r, a, t):
    ux = np.cos(theta)
    uy = np.sin(theta)
    return r * (ux * np.sin(t + a) - uy * np.cos(t + a) - t *( ux*np.cos(t + a) + uy*np.sin(t + a))) + x0*uy - y0 * ux

def extrema(x0, y0, theta, rb, a, t):
    ux = np.cos(theta)
    uy = np.sin(theta)
    C = x0*uy - y0*ux
    return C - rb * t * (ux*np.cos(t + a) + uy*np.sin(t + a))

# def derivative2(theta, rb, a, t):
#     ux = np.cos(theta)
#     uy = np.sin(theta)
#     return rb * (ux * np.sin(t + a) - uy * np.cos(t + a) - t *( ux*np.cos(t + a) + uy*np.sin(t + a)))

# Plot a single involute curve
t = np.linspace(0, 5*np.pi, 1000)
x, y = involuter(1, t, 0.0)


# Test point
x0 = -2.0
y0 = 10.0
theta = 0.33*np.pi
d = 13.0

r0 = np.sqrt(x0**2 + y0**2)


fig, ax = plt.subplots(3,1, figsize=(8,10))
ax[0].plot(x, y, label="involute")
ax[0].arrow(x0, y0, np.cos(theta)*d, np.sin(theta)*d, width=0.01, color='r', head_length=0.1, head_width=0.1, length_includes_head=True)


ax[0].add_artist(plt.Circle((0, 0), 1, fill=False))
ax[0].add_artist(plt.Circle((0, 0), r0, fill=False, color='r'))
ax[0].set_aspect('equal')
ax[0].grid(True)


t_0 = np.sqrt(r0**2/1.0 - 1)

# Calculate the extrema in terms of parameter t
te = np.array([np.arctan(np.sin(theta)/np.cos(theta)) + np.pi*i for i in range(6)])
te = te[te > 0.0]

ax[1].plot(t, intesection_curve(x0, y0, theta, 1.0, 0.0, t))
ax[1].plot(t_0, 0.0, 'o', label="T at start point")
ax[1].plot(te,  intesection_curve(x0, y0, theta, 1.0, 0.0, te), 'o', label="T at extrema", color='g')
ax[1].grid(True)

ax[1].plot(te, extrema(x0, y0, theta, 1.0, 0.0, te), 'o', label="extrema", color='k')

ax[1].vlines([np.pi*i  for i in range(6)], -10, 10, linestyles='dashed', color='k')


# Plot the extrema points on the involute curve
ax[0].plot( *involuter(1, te, 0.0), 'o', label="extrema", color='g')

# Plot tangent line
x_co = np.linspace(-6, 6, 1000)
ax[0].plot(x_co, np.tan(theta + 0.5*np.pi)*(x_co), label="tangent line", color='m', linestyle='dashed', linewidth=0.5)


##### Calculate projections on the tangent line
u_t = np.array([np.cos(theta - 0.5*np.pi), np.sin(theta - 0.5*np.pi)])

ax[2].plot(np.dot(np.array([x0, y0]), u_t), 0.0, 'o', label="start point", color='orange')

xe, ye = involuter(1, te, 0.0)

p_e = np.vstack((xe, ye)).T

ax[2].plot(np.matmul(p_e, u_t), np.zeros(len(xe)), 'o', label="extrema", color='k')



plt.show()
