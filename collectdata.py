import os
import cv2
import time
import pygame
import numpy as np
import pandas as pd
from getjoy import joy_check
from getkeys import key_check
from grabscreen import grab_screen
import win32gui, win32ui, win32con, win32api


# initialise pygame to collect joystick inputs
pygame.init()
pygame.joystick.init()

# list of keys that will be checked
keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


# function to collect the joystick inputs
# we need throttle, RT or z-axis
# and steering, x-axis
def joy_check():
    pygame.event.get()
    joystick_count = pygame.joystick.get_count()
    joystick = pygame.joystick.Joystick(1)
    joystick.init()
    axisX = joystick.get_axis( 0 )
    axisZ = joystick.get_axis( 2 )
    return round(axisX,4),round(axisZ,4)


# function the collect the keyboard inputs
def key_check():
    keys = []
    for key in keyList:
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


# function to grab a screenshot 
# at the mentioned coordinates
def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def main():

    # initialise empty list to which 
    # we will append the training data
    training_data = []

    paused = False

    # count down
    print('Data Collection Starts in')
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    # index of the first data point
    i = 0

    while(True):

        if not paused:

            # assign name and path of the image file
            image_name = str(i) + '.jpg'
            path = 'data\\IMG1\\' + image_name

            # take a screen shot of the game window
            # in our case it is in the top left of the screen
            # hence the coordinates (0,40) and (800,600)
            screen = grab_screen(region = (0,40,800,600))

            # get the corresponding joystick values
            steering,throttle = joy_check()

            # convert the colour profile of the image and
            # save it in the above assigned path
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, screen)

            # append the image patha dn joystick values collected
            training_data.append([image_name,steering,throttle])

            # after every 1000 data points collected
            # print length of training data
            if len(training_data) % 1000 == 0:
                print(len(training_data))

                # if total number of data points
                # collected is 100,000, save training data
                # and stop collection 
                if len(training_data) == 100000:
                    np.save('data\\training_data_4.npy',training_data)
                    print('SAVED')
                    break

            i += 1
        
        # if key 'T' s pressed
        # pause collection of data          
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

    print('----------------------------------------Collected----------------------------------------')


if __name__ == '__main__':
	main()