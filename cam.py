import numpy as np
from matplotlib import pyplot as plt

from skimage import io, transform, feature, util, color

from simple_pid import PID

import time
import random

#self-defined
from callBash import *

'''
requires gray image
'''
def getCoords(img):
    try:
        if (detectorNum == 0):
            #int coords
            print(f'Using Harris Corner Detector.')
            coords = feature.corner_peaks(feature.corner_harris(img), min_distance=5)
        elif (detectorNum == 1):
            print(f'Using Kitchen and Rosenfeld corner measure.')
            coords = feature.corner_peaks(feature.corner_kitchen_rosenfeld(img))
        elif (detectorNum == 2):
            print(f'Using Moravec corner measure.')
            coords = feature.corner_peaks(feature.corner_moravec(img))

        #subpix coords
        coords_subpix = feature.corner_subpix(img, coords, window_size=13)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
        ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
        plt.show()
        #append to log.dat
        #appendFile(f'get corner coords', 0)

        #print coords
        print(f'get corner coords:')
        print(f'coords:')
        print(f'{coords}\n')
        print(f'subpix coords:')
        print(f'{coords_subpix}\n')
        #for i in range(coords.size):
            #print(f'r{i}={coords[i][0]}; c{i}={coords[i][1]}')
        return (coords, coords_subpix)
    except:
        print(f'Error. Corner detection failed.')

'''
t, b can be array for coords or subpix-coords
'''
def clearTopBottom(t, b):
    t.clear()
    b.clear()

'''
coords = all corners
first four coords cooresponds to top square
last four coords cooresponds to bottom square
t, b can be array for coords or subpix-coords
'''
def setTopBottom(coords, t, b):
    #clear and append to avoid datatype error
    clearTopBottom(t, b)
    length = coords.shape[0]
    if length==8:
        for i in range(0,4):
            t.append((coords[i][0], coords[i][1]))
        for i in range(4,8):
            b.append((coords[i][0], coords[i][1]))
    else:
        print(f'Warning: Detected coordinates not 8. Spliting evenly into top and bottom.')
        for i in range(0,int(length/2)):
            t.append((coords[i][0], coords[i][1]))
        for i in range(int(length/2), length):
            b.append((coords[i][0], coords[i][1]))
    #####DEBUG#####
    print(f'{t}\n{b}\n')

'''
Verify coordinates do not specify white background by checking midpoint of centroid of first 3 coords and last 3 coords.
expect input to be top or bottom only. len(top) == len(bottom) == 4
img requires hsv
'''
def checkNotWhite(tb, img):
    #print(f'{tb}')
    #check if coords is of size 4
    if len(tb) != 4:
        print(f'Warning: length of array not 4.')

    #centroid of first 3 coords
    print(f'Using first 3 coords: {tb[0]}; {tb[1]}; {tb[2]}.')
    r1 = (tb[0][0] + tb[1][0] + tb[2][0])/3
    c1 = (tb[0][1] + tb[1][1] + tb[2][1])/3
    print(f'Centorid of first 3 coords: ({r1}, {c1})')
    #centroid of last 3 coords
    print(f'Using last 3 coords: {tb[-1]}; {tb[-2]}; {tb[-3]}.')
    r2 = (tb[-1][0] + tb[-2][0] + tb[-3][0])/3
    c2 = (tb[-1][1] + tb[-2][1] + tb[-3][1])/3
    print(f'Centorid of last 3 coords: ({r2}, {c2})')
    #calculate midpoint of two centroids
    r = int((r1 + r2)/2)
    c = int((c1 + c2)/2)
    print(f'Midpoint of centroids, converted to integers: ({r}, {c})')

    #check if S!=0 aka is not white
    if img[r][c][1] != 0:
        return True
    else:
        return False

'''
expect len(tb)==4
angle1 is angle from first two coords in tb
angle2 is angle from last two coords in tb

ang>0 if CCW,
amg<0 if CW

return avg of angle1 and angle2
'''
def findRotAngle(tb):
    angle1 = np.arctan(abs(tb[0][0] - tb[1][0]) / abs(tb[0][1] - tb[1][1]))
    angle2 = np.arctan(abs(tb[2][0] - tb[3][0]) / abs(tb[2][1] - tb[3][1]))
    ang = (angle1 + angle2) / 2
    if (tb[0][1] > tb[3][1]) and (tb[2][1] > tb[1][1]):
        ang = -np.rad2deg(ang)
        print(f"CW direction: {ang} degrees.")
        return ang
    elif (tb[0][1] < tb[3][1]) and (tb[2][1] < tb[1][1]):
        ang = np.rad2deg(ang)
        print(f"CCW direction: {ang} degrees.")
        return ang

'''
Set up
'''

#initiate pid
pid = PID(1.0, 0.00, 0.00, setpoint=0, sample_time=None, output_limits=(-30, 30))

#array to record angle needed and pid output values
val=[]

#array to record simulated noise
#initiated with 0 cuz no noise in the initial measurements.
noise=[0]

#corner detector flags
detectorNum = 0
'''
harris = 0, rosenfeld = 1, moravec = 2
'''

#initiate empty top and bottom arrays
top = []
bottom = []

#initiate empty topSubpix and bottomSubpix arrays
topSubpix = []
bottomSubpix = []

'''
File for simulation
'''
filename = "squares_colour.png"

#boolean flag for initial reading
initial = True

'''
Main loop starts here
'''
while True:
    clear()
    #######
    #take picture
    #by transforming the test picture directly this part is omitted and awaits hardware
    #######

    #######
    #read image
    print(f'Reading image...')
    #if test image has no alpha
    #img_rgb = io.imread(filename)
    
    '''
    Read from file for initial scanning of printbed.
    Add pure rotation to file for simulation purposes
    '''
    if initial:
        #if test image has alpha channel
        img_rgb = color.rgba2rgb(io.imread(filename))
        img_rgb = transform.rotate(img_rgb, 10, cval=1, mode='constant')
        initial = False
    
    #####DEBUG#####
    #plt.imshow(img_rgb)
    #plt.show()

    #get gray image
    img_gray = color.rgb2gray(img_rgb)
    
    #get hsv image
    img_hsv = color.rgb2hsv(img_rgb)
    print(f'Image ready.')
    #######

    #######
    #get corner coordiantes
    print(f'Getting corner coordinates...')
    #get coordinates for detected corners.
    #getCoords require gray image
    (coords, subpix) = getCoords(img_gray)
    #print(f'coords.size={coords.shape}')
    #######

    #######
    #set coordinate arrays
    setTopBottom(coords, top, bottom)
    setTopBottom(subpix, topSubpix, bottomSubpix)
    #top = [ (r0, c0), (r1, c1), (r2, c2), (r3, c3) ]
    #bottom = [ (r0, c0), (r1, c1), (r2, c2), (r3, c3) ]
    #######

    #######
    #check if 8 corners are detected.
    if (len(top)<4) or (len(bottom)<4):
        print(f'Warning: Not all corners detected.')
        print(f'Skip to next loop using another detector.')
        detectorNum = (detectorNum + 1) % 3 #harris = 0, rosenfeld = 1, moravec = 2
        #skip to next loop if not all detected
        #rand = random.uniform(-1,1)
        #noise.append(rand)
        #print(f'Noise: {rand}')
        #change = rand
        #img_rgb = transform.rotate(img_rgb, change, cval=1, mode='constant')
        time.sleep(2)
        continue
    elif (len(top)>4) or (len(bottom)>4):
        print(f'Warning: More than 8 corners detected.')
        print(f'Skip to next loop using another detector.')
        detectorNum = (detectorNum + 1) % 3 #harris = 0, rosenfeld = 1, moravec = 2
        time.sleep(2)
        continue
    #######

    #######
    #check top centroid not white
    try:
        #checkNotWhite requires hsv
        if not checkNotWhite(top, img_hsv):
            print(f'Warning: Top shape is not white.')
    except:
        print(f'Error: Cannot check if top shape is not white.')

    try:
        #check bottom centroid not white
        #checkNotWhite requires hsv
        if not checkNotWhite(bottom, img_hsv):
            print(f'Warning: Bottom shape is not white.')
    except:
        print(f'Error: Cannot check if bottom shape is not white.')
    #######

    #######
    #find rotation angle
    try:
        print(f'Calculating rotation angle...')
        angleTop = findRotAngle(topSubpix)
        print(f'Top angle: {angleTop:.6f}')
    except:
        angleTop = 0
        print(f'Warning: Cannot find top rotation angle. Default to 0 degrees.')

    try:
        angleBottom = findRotAngle(bottomSubpix)
        print(f'Bottom angle: {angleBottom:.6f}')
    except:
        angleBottom = 0
        print(f'Warning: Cannot find bottom rotation angle. Default to 0 degrees.')

    angle = ( angleTop + angleBottom ) / 2
    print(f'Average angle: {angle:.6f}')
    #######


    #######
    #end loop condition
    if (angle > -0.5) and (angle < 0.5):
        print(f'Done.')
        break
    else:
        #######
        #get new output from pid
        '''
        NOTE:
        angle obtained above is angle NEEDED to FIX THE ERROR
        error is -angle NOT angle
        '''
        control = pid(-angle)
        val.append((angle, control))
        print(f'PID: {control:.6f}')
        print(f'{val}')
        #######

        #update printbed rotation
        print(f'Updating printbed...')
        '''
        Extra random rotation in [-0.25, 0.25] as random noise
        '''
        rand = random.uniform(-0.25,0.25)
        noise.append(rand)
        print(f'Noise: {rand}')
        change = control + rand

        img_rgb = transform.rotate(img_rgb, change, cval=1, mode='constant')

        time.sleep(2)
    ######

print(f'{val}')
print(f'{noise}')
