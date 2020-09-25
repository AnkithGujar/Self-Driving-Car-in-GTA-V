import cv2
import sys
import time
import utils
import random
import numpy as np
from getkeys import key_check
from statistics import mode,mean
from grabscreen import grab_screen
from keras.models import load_model
from darkflow.net.build import TFNet
from collections import deque, Counter
from vjoy import vJoy, ultimate_release


def main():

    vj = vJoy()

    paused = False

    XYRANGE = 16393
    ZRANGE = 32786
    wAxisX = 16393
    wAxisZ = 0
    wAxisZRot = 0

    options = {"model": "models/yolov2-tiny.cfg", "load": "models/yolov2-tiny.weights", "threshold": 0.1, "gpu": 0.5}
            
    tfnet = TFNet(options)

    model = load_model("models/model_gtav.h5")

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    while(True):
        
        if not paused:

            last_time = time.time()

            screen = grab_screen(region = (0,40,800,600))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            overlay = screen.copy()

            result = tfnet.return_predict(screen)
            for box in result:
                if box['confidence'] > 0.5:
                    pt1 = (box['topleft']['x'],box['topleft']['y'])
                    pt2 = (box['bottomright']['x'],box['bottomright']['y'])
                    cv2.rectangle(screen, pt1, pt2, color = (255,255,255), thickness = 2, lineType = 0, shift = 0)
                    cv2.putText(screen, '{}'.format(box['label']), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                    apx_dst = (box['bottomright']['x'] - box['topleft']['x']) / 800
                    mid_x = ((box['bottomright']['x'] + box['topleft']['x']) / 2) / 800
                    if apx_dst >= 0.2:
                        if mid_x > 0.35 and mid_x < 0.65:
                            cv2.rectangle(overlay, pt1, pt2, color = (0,0,255), thickness = -1, lineType = 0, shift = 0)
                            screen = cv2.addWeighted(overlay, 0.4, screen, 0.6, 0)
                            cv2.putText(screen, 'WARNING', pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)
            
            cv2.imshow('image',screen)
            cv2.waitKey(1)

            image = np.asarray(screen)
            image = utils.preprocess(image)
            image = np.array([image])

            outputs = model.predict(image, batch_size = 1)

            turn = float(outputs[0])
            throttle = 0.5

            fps  = 1 / round(time.time()-last_time, 3)
            
            if throttle > 0:

                vj.open()
                joystickPosition = vj.generateJoystickPosition(wAxisZRot = int((ZRANGE*throttle)),wAxisX = int(XYRANGE + (turn*XYRANGE)))
                vj.update(joystickPosition)
                time.sleep(0.001)
                vj.close()
                print('FPS {}. Steer: {}. Throttle: {}. Brake: {}'.format(fps, turn, throttle, 0))  

            else:

                vj.open()
                joystickPosition = vj.generateJoystickPosition(wAxisZ = int(-1*(ZRANGE*throttle)),wAxisX = int(XYRANGE + (turn*XYRANGE)))
                vj.update(joystickPosition)
                time.sleep(0.001)
                vj.close()
                print('FPS {}. Steer: {}. Throttle: {}. Brake: {}'.format(fps, turn, 0, throttle))  
                    

        keys = key_check()
        if 'T' in keys:
            ultimate_release()
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


if __name__ == '__main__':

    main()       
