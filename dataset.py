# Import the necessary packages
import argparse
from imutils import paths
import cv2
import os.path
import matplotlib
import matplotlib.pyplot as plt
import pprint
import numpy as np
import subprocess

# Define input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input data")
ap.add_argument("-o", "--output", required=True,
                help="path to output")
args = vars(ap.parse_args())

input_dir = args["input"]
letters_dir = args["output"]

imagePaths = list(paths.list_images(input_dir))
counts = {}

key_name = []


def press(event):
    if event.key == '`':
        print("[INFO] IGNORE")
    else:
        print("Pressed: ", event.key)
        key_name.append(str(event.key))
    plt.close()


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


file_list = []

# Looping over images in the path (imagePath \in imagePaths)
for (i, imagePath) in enumerate(imagePaths):

    debug_num = -1  # put number here to start from a certain iteration
    if i < debug_num:
        continue

    print(i)
    # grab the base filename as the text
    file_name = os.path.basename(imagePath)
    file_list.append(file_list)
    file_name_path = os.path.join(input_dir, file_name)
    captcha_text = os.path.splitext(file_name)[0]
    print("[{}/{}] Images Processed".format(i + 1, len(imagePaths)) + " | Current: " + file_name)

    try:
        image = cv2.imread(imagePath)  # reading the single image
        image2 = image.copy()
        (h, w, d) = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        simple_captcha = 0
        if simple_captcha == 1:
            gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        else:
            _, mask = cv2.threshold(gray, 254 / 2 + 100, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            black = np.zeros_like(image)
            black2 = black.copy()
            blank_image = np.zeros((h, w, d), np.uint8)
            blank_image[:] = (255, 255, 255)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Smoothening is done with the help of Gaussian Blur. To do so, image convolution technique is applied with a
        # Gaussian Kernel (3x3, 5x5, 7x7 etc…). The kernel size depends on the expected blurring effect. Basically,
        # the smallest the kernel, the less visible is the blur.

        # Image thresholding is an important intermediary step for image processing pipelines. Thresholding can help
        # us to remove lighter or darker regions and contours of images. CV_THRESH_BINARY,  CV_THRESH_OTSU is a
        # required flag to perform Otsu thresholding. Because in fact we would like to perform binary thresholding,
        # so we use CV_THRESH_BINARY (you can use any of 5 flags opencv provides) combined with CV_THRESH_OTSU
        retVal, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        newRetVal = retVal * 0.7

        _, new_thresh = cv2.threshold(blurred, newRetVal, 255, cv2.THRESH_BINARY_INV)

        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
        opening = cv2.morphologyEx(new_thresh, cv2.MORPH_CLOSE, kernel)

        # Repair text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        dilate = cv2.dilate(opening, kernel, iterations=1)

        # Bitwise-and with input image
        result = cv2.bitwise_and(image, image, mask=dilate)
        result[dilate == 0] = (255, 255, 255)

        # Store images
        image_attributes = []

        # Store contour x, y, w, h so we can sort it
        contour_boxes = []
        contours, conts_hierarchy = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        index = 0

        debug = 1
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
            # cv2.drawContours(image2, [hull], -1, (0, 0, 255), 1)
            # extract character from current contour
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)

            masked = cv2.bitwise_and(new_thresh, mask)
            x, y, w, h = cv2.boundingRect(masked)  # should be masked

            # extract only the character
            extracted_cha = 255 - masked[y:y + h, x:x + w]

            if simple_captcha == 0:
                # print(x, y, w, h)
                # print(extracted_cha.shape[1], extracted_cha.shape[0])
                # shape[1] = w || shape[0] = h

                if w > 94:  # four letters width
                    fourth_width = int(extracted_cha.shape[1] / 4)
                    image_attributes.append(extracted_cha[:, :fourth_width])  # 1 -> 1/4
                    image_attributes.append(extracted_cha[:, fourth_width:fourth_width * 2])  # 1/4 -> 2/4
                    image_attributes.append(extracted_cha[:, fourth_width * 2:fourth_width * 3])  # 2/4 -> 3/4
                    image_attributes.append(extracted_cha[:, fourth_width * 3:fourth_width * 4])  # 3/4 -> 4/4
                    contour_boxes.append((x, y, fourth_width, h))
                    contour_boxes.append((x + fourth_width, y, fourth_width, h))
                    contour_boxes.append((x + 2 * fourth_width, y, fourth_width, h))
                    contour_boxes.append((x + 3 * fourth_width, y, fourth_width, h))

                elif w > 66:  # three letters width
                    third_width = int(extracted_cha.shape[1] / 3)
                    image_attributes.append(extracted_cha[:, :third_width])
                    image_attributes.append(extracted_cha[:, third_width:third_width * 2])
                    image_attributes.append(extracted_cha[:, third_width * 2:third_width * 3])
                    contour_boxes.append((x, y, third_width, h))
                    contour_boxes.append((x + third_width, y, third_width, h))
                    contour_boxes.append((x + 2 * third_width, y, third_width, h))
                elif w > 35:  # two letter width
                    half_width = int(extracted_cha.shape[1] / 2)
                    image_attributes.append(extracted_cha[:, :half_width])
                    image_attributes.append(extracted_cha[:, half_width:half_width * 2])
                    contour_boxes.append((x, y, half_width, h))
                    contour_boxes.append((x + half_width, y, half_width, h))
                else:
                    image_attributes.append(extracted_cha)
                    contour_boxes.append((x, y, w, h))
                pass
            elif simple_captcha == 1:
                if w / h > 1.25:
                    # This contour is too wide to be a single letter!
                    # Split it in half into two letter regions!
                    half_width = int(w / 2)
                    image_attributes.append((x, y, half_width, h))
                    image_attributes.append((x + half_width, y, half_width, h))
                else:
                    # This is a normal letter by itself
                    image_attributes.append((x, y, w, h))

            cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if debug == 1:
            f, axarr = plt.subplots(6, 1, figsize=(10, 20))
            f.canvas.mpl_connect('key_press_event', press)
            for ax in axarr:
                ax.set_xticks([])
                ax.set_yticks([])
            # use the created array to output your multiple images. In this case I have stacked 4 images vertically
            axarr[0].imshow(thresh, cmap='gray')
            axarr[1].imshow(new_thresh, cmap='gray')
            axarr[2].imshow(opening, cmap='gray')
            axarr[3].imshow(result, cmap='gray')
            axarr[4].imshow(image2, cmap='gray')
            axarr[5].imshow(extracted_cha, cmap='gray')
            plt.show()
        else:
            p = subprocess.Popen(["display", file_name_path])

        sorted_characters = []
        zipped_info = sorted(zip(contour_boxes, image_attributes))
        for item_a, item_b in zipped_info:
            sorted_characters.append(item_b)

        # Save each letter as a single image for wikipedia
        if simple_captcha == 1:
            # Save out each letter as a single image
            for letter_box, letter_text in zip(image_attributes, captcha_text):
                # Grab the coordinates of the letter in the image
                x, y, w, h = letter_box

                # Extract the letter from the original image with a 2-pixel margin around the edge
                letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

                # Get the folder to save the image in
                save_path = os.path.join(letters_dir, letter_text)

                # if the output directory does not exist, create it
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # write the letter image to a file
                count = counts.get(letter_text, 1)
                p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
                cv2.imwrite(p, letter_image)

                # increment the count for the current key
                counts[letter_text] = count + 1
        elif simple_captcha == 0:
            for letters in sorted_characters:
                # Grab the coordinates of the letter in the image

                # Extract the letter from the original image with a 2-pixel margin around the edge
                # letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
                if debug == 0:
                    fig, ax = plt.subplots()
                    ax.imshow(letters, cmap='gray')
                    plt.xticks([]), plt.yticks([])
                    move_figure(fig, 250, 300)

                    fig.canvas.mpl_connect('key_press_event', press)
                    plt.show()

                    key_file = key_name.pop()
                    dirPath = os.path.join(letters_dir, key_file)
                    # print(dirPath)

                    if not os.path.exists(dirPath):
                        os.makedirs(dirPath)
                    count = counts.get(key_file, 1)
                    filePath = os.path.join(dirPath, "{}.png".format(str(count).zfill(6)))
                    print(filePath)

                    fig.savefig(filePath)

                    counts[key_file] = count + 1

        if debug == 0:
            p.kill()

    except KeyboardInterrupt:
        print("[INFO] manually leaving script at:", file_name)
        plt.close()
        pprint(file_list)
        break
