# import the necessary packages
import argparse
import os
from pickle import load
from helpers import resize_to_fit
import matplotlib.pyplot as plt
import numpy as np
import cv2
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from helpers import add_padding

# Set the random seed
from helpers import preprocess

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of images")
ap.add_argument("-m", "--trained_model", required=True,
                help="path to input trained_model")
args = vars(ap.parse_args())

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["trained_model"])
lb = load(open(os.path.join(args["trained_model"], "label_binarizer.pkl"), 'rb'))
# randomly sample a few of the input images
imagePaths = list(paths.list_images(args["input"]))
# imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

# loop over the image paths
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    padded = add_padding(image)

    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    simple_captcha = 0
    if simple_captcha == 1:
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    temp, temp_thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = 0.8 * temp
    _, new_thresh = cv2.threshold(gray, temp, 255, cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 7))
    threshed = cv2.morphologyEx(new_thresh, cv2.MORPH_CLOSE, rect_kernel)

    # Store images
    image_attributes = []
    contour_boxes = []

    contours, conts_hierarchy = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initialize the output image as a "grayscale" image with 3
    # channels along with the output predictions
    output = cv2.merge([gray] * 3)
    predictions = []
    for contour in contours:
        # x, y = coordinates for the top left corner of the box
        # w, h = width and height
        area = cv2.contourArea(contour)

        if simple_captcha == 0:
            if area < 10:  # this catches dots and ignore them, we don't want anything smaller than area of 10
                continue
        else:
            if area < 3:
                continue

        hull = cv2.convexHull(contour)
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y, w, h)

        if simple_captcha == 0:
            if w > 94:  # four letters width
                one_fourth_width = int(w / 4)
                image_attributes.append((x, y, one_fourth_width, h))
                image_attributes.append((x + one_fourth_width, y, one_fourth_width, h))
                image_attributes.append((x + 2 * one_fourth_width, y, one_fourth_width, h))
                image_attributes.append((x + 3 * one_fourth_width, y, one_fourth_width, h))
            elif w > 62:  # three letters width
                one_third_width = int(w / 3)
                image_attributes.append((x, y, one_third_width, h))
                image_attributes.append((x + one_third_width, y, one_third_width, h))
                image_attributes.append((x + 2 * one_third_width, y, one_third_width, h))
            elif w > 35:  # two letter width
                half_width = int(w / 2)
                image_attributes.append((x, y, half_width, h))
                image_attributes.append((x + half_width, y, half_width, h))
            else:
                image_attributes.append((x, y, w, h))
        else:
            if w / h > 1.25:
                # This contour is too wide to be a single letter!
                # Split it in half into two letter regions!
                half_width = int(w / 2)
                image_attributes.append((x, y, half_width, h))
                image_attributes.append((x + half_width, y, half_width, h))
            else:
                # This is a normal letter by itself
                image_attributes.append((x, y, w, h))

    for letter_box in image_attributes:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_box
        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter_image, 28, 28)

        """
        roi = preprocess(letter_image, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = trained_model.predict(roi).argmax(axis=1)[0]
        """
        # Ask the neural network to make a prediction
        if simple_captcha == 1:
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)
            prediction = model.predict(letter_image)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter = lb.inverse_transform(prediction)[0]
        else:
            roi = preprocess(letter_image, 28, 28)
            roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
            pred = model.predict(roi).argmax(axis=1)[0]
            letter = lb.classes_[pred]

        predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2),
                      (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, str(letter), (x - 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    plt.title(os.path.basename(imagePath))
    plt.imshow(output, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show(block=True)

    print("[INFO] captcha: {}".format("".join(predictions)))
