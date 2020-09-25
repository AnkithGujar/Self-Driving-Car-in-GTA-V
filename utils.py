import cv2, os
import numpy as np
import matplotlib.image as mpimg


# initialise the dimensions of the image
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 270, 480, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


# given the input data, yields batches of it to the training loop
def batch_generator(image_paths, output_values, batch_size, is_training):
    
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    outputs = np.empty(batch_size)

    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center = image_paths[index]
            steering_angle = output_values[index]

            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(center, steering_angle)
            else:
                image = load_image(center) 
            
            images[i] = preprocess(image)
            outputs[i] = steering_angle
            # outputs[i,1] = throttle_value

            i += 1

            if i == batch_size:
                break
        yield images, outputs


# calls all the random augmentation functions
def augument(center, steering_angle, range_x=100, range_y=10):
    image, steering_angle = load_image(center), steering_angle
    image = resize(image)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


# perform random horizontal flip to the image
def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


# perform rndom translation to the image
def random_translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


# apply random shadows to the image
def random_shadow(image):
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    mask = np.zeros_like(image[:, :, 1])
    a = (ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1)
    mask[a > 0] = 1
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


# change the brightness of the image randomly
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# function to preprocess the image
def preprocess(image):
    image = crop(image)
    image = resize(image)
    return image


# function to load the image from the mentioned path
def load_image(image_file):
    return mpimg.imread('data\\IMG2\\' + image_file)


# function the crop the bottom of the image to remove the hood of the car
def crop(image):
    return image[60:-25, :, :]


# function to resize the images to the specified dimensions
def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)