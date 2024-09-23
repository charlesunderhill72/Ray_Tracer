#!/usr/bin/env python3

#
# ray_tracer.py - Ray tracing algorithm to render height field of a fractal.
#
# 19Jul21 Charles Underhill
#

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

### Necessary Functions ###

def compute_mandelbrot(X, Y, x0, x1, y0, y1, iterations):
   """
   X and Y define the resolution. x0, x1, y0, and y1 define the desired 
   boundaries of the mandelbrot set to compute. Iterations is the desired # 
   of iterations to test for convergence. Returns array of color values 
   representing stability of complex numbers in the desired range.
   """

   x_range = np.linspace(x0, x1, X)
   y_range = np.linspace(y0, y1 , Y)
   c_x, c_y = np.meshgrid(x_range, 1j*y_range)
   c = c_x + c_y
   z = np.zeros_like(c)
   max_it = iterations
   pvals = np.zeros_like(z, dtype='uint')

   for i in range(max_it+1):
      diverges = abs(z) > 2
      stable = abs(z) < 2
      z = z**2 + c
      pvals[diverges] = pvals[diverges] + 1
      pvals[stable] = 0


   return pvals


###############################################################################


def normalize(varray):
   """
   Takes array of 3D vectors as its input and normalizes each vector. 
   Returns array of normalized vectors.
   """
   rows = np.shape(varray)[0]
   columns = np.shape(varray)[1]
   narray = np.zeros((rows, columns, 3))
   for i in range(rows):
      for j in range(columns):
         narray[i][j] = varray[i][j] / np.linalg.norm(varray[i][j])

   return narray


###############################################################################


def surface_normal(x_range, y_range, z_grid):
   """
   Takes range of x and y values with corresponding grid of z surface values 
   as input and computes surface normals at each point.
   """
   x_size = len(x_range)
   y_size = len(y_range)
   normals = np.zeros((x_size, y_size, 3))
   for i in range(x_size):
      for j in range(y_size):
         try:
            dx = z_grid[i][j] - z_grid[i+1][j]      # Try and except used to 
            dy = z_grid[i][j-1] - z_grid[i][j]      # avoid IndentationError.
            dz = 1 / np.sqrt(x_size * y_size)
            r = np.sqrt(dx**2 + dy**2 + dz**2)
         except:
            dx, dy, dz, r = 0, 0, 1, 1              # I set normals on the 
         normals[i][j] = (dx / r, dy / r, dz / r)   # edges to (0, 0, 1) so
                                                    # it looks flat.
   return normals


##############################################################################


def Intensity(cos_theta, cos_alpha, ambient, diffuse, spectral, shininess):
   """
   Intensity function for simple light model. Takes in array of cos_theta 
   values where theta is the angle between light ray and surface normal
   for every point on the surface. Alpha is the angle between the view 
   vector and reflected light vector for every point.

   Returns array of light intensities for every point on a surface.
   """
   I_vals = np.zeros(np.shape(cos_theta))
   less = cos_theta < 0
   other = cos_theta >= 0
   I_vals[less] = ambient
   I_vals[other] = I_vals[other] + ambient + diffuse * cos_theta[other] + spectral * (cos_alpha[other])**shininess
   return I_vals


##############################################################################


intro = """
\nWelcome to my Mandelbrot Set ray tracing program. In order to render 
3D images of the Mandelbrot set, you will have to specify the region 
of the set you would like to compute.
"""

intro1 = """
\nNow you must specify the number of data points along the x and y 
axis to compute. This will be the same for the x and y axis. 
(Recommended 300 for fast computing time).
"""

intro2 = """
\nLastly, you must specify the angle from the ground that you would 
like to view the image from, and the height above the ground that you 
would like the light to be located at. The x and y values of the light 
are automatically set to the middle of the specified range in order to 
ensure the scene receives some light.
"""

print(intro)

while True:
   instr = input("\nPlease enter the lower bound for the x (real) axis: ")
   try:
      x0 = float(instr)
   except ValueError:
      print("\nYour input was not a float.  Try again.\n", file=sys.stderr)
   else:
      break

while True:
   instr = input("\nPlease enter the upper bound for the x (real) axis: ")
   try:
      x1 = float(instr)
   except ValueError:
      print("\nYour input was not a float.  Try again.\n", file=sys.stderr)
   else:
      break

while True:
   instr = input("\nPlease enter the lower bound for the y (imaginary) axis: ")
   try:
      y0 = float(instr)
   except ValueError:
      print("\nYour input was not a float.  Try again.\n", file=sys.stderr)
   else:
      break

while True:
   instr = input("\nPlease enter the upper bound for the y (imaginary) axis: ")
   try:
      y1 = float(instr)
   except ValueError:
      print("\nYour input was not a float.  Try again.\n", file=sys.stderr)
   else:
      break

print(intro1)

while True:
   instr = input("\nPlease enter number of data points to compute: ")
   try:
      N = int(instr)
   except ValueError:
      print("\nYour input was not an integer.  Try again.\n", file=sys.stderr)
   else:
      break


print(intro2)

while True:
   instr = input("\nPlease enter the light height (values between 0 and 2 are recommended): ")
   try:
      light_height = float(instr)
   except ValueError:
      print("\nYour input was not a float.  Try again.\n", file=sys.stderr)
   else:
      break


while True:
   instr = input("\n\nFinally, please enter the view angle in degrees (0 < ang < 90): ")
   try:
      deg_ang = float(instr)
      view_angle = (np.pi / 180) * deg_ang
      if view_angle <= 0 or view_angle >= np.pi/2:
         raise ValueError
   except ValueError:
      print("\nYour input was not a float or was not in the specified range.  Try again.\n", file=sys.stderr)
   else:
      break



scale_factor = 2
light = np.array([(x0 + x1) / 2, (y0 + y1) / 2,  light_height])
view_vector = np.array([0, -1*np.cos(view_angle), np.sin(view_angle)])

print()
print('\n\nComputing Mandelbrot Set in the specified range...')
print()

t0 = time.perf_counter()
Z = -1 * scale_factor * compute_mandelbrot(N, N, x0, x1, y0, y1, 500)
Z = np.interp(Z, (Z.min(), Z.max()), (0, 1))
t = time.perf_counter() - t0

print()
print('Time elapsed computing Mandelbrot Set: %f s' % t)
print()


x_vals = np.linspace(0, 1, N)
y_vals = np.linspace(0, 1, N)


# Surface normals are essential in computing the trajectory of light rays off 
# of the surface. This function aims to compute the surface normal for each 
# point p = (x_i, y_j, z_(i,j)).
 
print()
print('Computing surface normals...')
print()

t0 = time.perf_counter()
normals = surface_normal(x_vals, y_vals, Z)
normals = normalize(normals)
t = time.perf_counter() - t0

print()
print('Time elapsed computing surface normals: %f s' % t)
print()

# What I ultimately need is to compute the angle between the normal vector 
# and the vector pointing towards the light source at every point on the 
# surface. I will try to create an array of the light vectors and take the 
# dot product with the array of normal vectors.


# Array of points on the surface

points = np.zeros((N, N, 3))
cos_theta = np.zeros((N, N))

for i, x in enumerate(x_vals):
   for j, y in enumerate(y_vals):
      points[i][j] = (x, y, Z[i][j])
      

# Array of normalized vectors pointing from point on surface to light source.

print('\nComputing light ray trajectories...\n')
t0 = time.perf_counter()
l_vectors = points - light
l_vectors = normalize(l_vectors)

# I'm going to compute cos_theta rather than theta for each normal, because
# the relevant functions for this algorithm only call for cos_theta.

for i in range(N):
   for j in range(N):
      cos_theta[i][j] = np.vdot(l_vectors[i][j], normals[i][j])


# Now I can compute the reflected vector for each point on the surface.

r_vectors = np.zeros((N, N, 3))
for i in range(N):
   for j in range(N):
      r_vectors[i][j] = (2 * cos_theta[i][j]) * normals[i][j] - l_vectors[i][j]

r_vectors = normalize(r_vectors)

# The last thing I need to render images is the angle alpha between each 
# reflected vector and the view vector, which is the vector from the 
# environment to the screen. Just like with theta, computing cos_alpha will
# be more useful in the long run.

cos_alpha = np.zeros((N, N))
for i in range(N):
   for j in range(N):
      cos_alpha[i][j] = np.vdot(view_vector, r_vectors[i][j])

t = time.perf_counter() - t0
print('\nTime elapsed computing light ray trajectories: %f s\n' %t)


# Now everything is ready to start rendering. I'll start by assigning some
# arbitrary values to different components of light.

I_a = 0         # Ambient
f_d = 0.6       # Diffuse
f_s = 0.4       # Specular
b = 2           # Shininess of surface

# Output array of intensities on the screen. Needs to be at least width by 
# sqrt(2) * height in size becaue of the way the data is projected.

intens = np.zeros((N, int((np.cos(view_angle) + np.sin(view_angle)) * N))) 


print('\nComputing light intensities on the surface...\n')

t0 = time.perf_counter()
I = Intensity(cos_theta, cos_alpha, I_a, f_d, f_s, b)
t = time.perf_counter() - t0

print('\nTime elapsed computing light intensities on the surface: %f s\n' % t)

print('\nBegin rendering...\n')
t0 = time.perf_counter()


# Rendering algorithm using floating horizon technique.


for i in range(N):
   p0 = int((N - 1) * (y_vals[0] * np.sin(view_angle) + Z[i][0] * np.cos(view_angle)))
   intens[0][p0] = I[i][0]
   horizon = p0

   for j in range(1, N):
      p1 = int((N - 1) * (y_vals[j] * np.sin(view_angle) + Z[i][j] * np.cos(view_angle)))         

      if p1 > horizon:
         intens[i][p1] = I[i][j]
         p = p1 - 1

         while p > horizon:
            h = (p - p0) / (p1 - p0)
            intens[i][p] = (1-h) * I[i][j-1] + h * intens[i][p1]
            p = p - 1
         horizon = p1
         
      p0 = p1


t = time.perf_counter() - t0

print()
print('Time elapsed while rendering image: %f s' % t)
print()


plotarr = np.flipud(intens.T)
f1, ax1 = plt.subplots()
picture = ax1.imshow(plotarr, interpolation='none', cmap='inferno')
ax1.axis('off')
f1.show()

input("\nPress <Enter> to exit...\n")


















