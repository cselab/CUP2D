import numpy as np 
import math
import random

IC = 'dense' # [uniform / spikes / dense]
circles = 4
particle_radius = 0.01
inner_radius = 0.1
outer_radius = 0.4

particles_per_circle = 10  # for uniform case
particles_inner_circle = 7 # for dense case

xvel = 0.0

# check possibility
if outer_radius>=0.5-particle_radius:
    raise Exception('Outer circle out of domain')

# Assemble radii and check spacing
r = np.zeros((circles))
for i in range(circles):
    r[i] = inner_radius + i * (outer_radius-inner_radius)/(circles-1)
if r[1]-r[0]<= particle_radius:
    raise Exception('Circle spacing too low.')

# Write first few lines by default    
f = open("launchParticles.sh","w+")
f.write("# Reynolds 1000 :\n"
    "NU=${NU:-0.000015625}\n"
    "# Reynolds 9500 :\n"
    "#NU=${NU:-0.000001644736842}\n"
    'OPTIONS="-bpdx 50 -bpdy 50 -tdump 0.01 -nu ${NU} -CFL 0.1 -iterativePenalization 0 -tend 50 -poissonType cosine "\n'
    'OBJECTS="')

if IC == 'uniform' or IC == 'spikes':
    # Construct particle's angles, uniform case
    angle_step = 360/particles_per_circle
    theta_arr = np.zeros((particles_per_circle))
    theta_arr[0] = angle_step
    for i in range(particles_per_circle-1):
        theta_arr[i+1] = theta_arr[i] + angle_step

    if r[0]*((math.pi/180)*(theta_arr[1]-theta_arr[0])) <= 2*particle_radius:
        raise Exception('Particles too close together within their circle.')
    
    angle_offset = 0
    for c in range(circles):
        r_now = r[c]
        if IC == 'uniform':
            angle_offset = random.randint(-20, 20) 

        theta_degrees = theta_arr + angle_offset
        theta = (math.pi/180)*theta_degrees
        for p in range(particles_per_circle):
            theta_now = theta[p]
            if c == circles-1 and p == particles_per_circle-1:
                f.write("disk_radius=" + str(particle_radius) + "_xpos=" + str(0.5 + r_now*np.cos(theta_now)) + "_ypos=" + \
                str(0.5 + r_now*np.sin(theta_now)) + "_bForced=1_bFixed=0_xvel=" + str(xvel) + '"\n')
            else:
                f.write("disk_radius=" + str(particle_radius) + "_xpos=" + str(0.5 + r_now*np.cos(theta_now)) + "_ypos=" + \
                str(0.5 + r_now*np.sin(theta_now)) + "_bForced=1_bFixed=0_xvel=" + str(xvel) +"\n")

    f.write('source launchCommon.sh')
    f.close()

elif IC == 'dense':
    # Define constant perimetral spacing through circles
    L = inner_radius * (2*math.pi)/particles_inner_circle

    for c in range(circles):
        r_now = r[c]
        # Construct particle's angles, circle dependent case
        angle_step_now = L/r_now
        particles_now = int(2*math.pi/angle_step_now)
        theta_arr_now = np.zeros((particles_now))
        theta_arr_now[0] = angle_step_now
        for i in range(particles_now-1):
            theta_arr_now[i+1] = theta_arr_now[i] + angle_step_now
        theta_arr_now = theta_arr_now + random.randint(-20,20)*(math.pi/180)

        for p in range(particles_now):
            theta_now = theta_arr_now[p]
            if c == circles-1 and theta_now == theta_arr_now[-1]:
                f.write("disk_radius=" + str(particle_radius) + "_xpos=" + str(0.5 + r_now*np.cos(theta_now)) + "_ypos=" + \
                str(0.5 + r_now*np.sin(theta_now)) + "_bForced=1_bFixed=0_xvel=" + str(xvel) + '"\n')
            else:
                f.write("disk_radius=" + str(particle_radius) + "_xpos=" + str(0.5 + r_now*np.cos(theta_now)) + "_ypos=" + \
                str(0.5 + r_now*np.sin(theta_now)) + "_bForced=1_bFixed=0_xvel=" + str(xvel) +"\n")

    f.write('source launchCommon.sh')
    f.close()