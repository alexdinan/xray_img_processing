# import required modules
import cv2
import argparse
import os
import sys
import numpy as np
from inpainter import Inpainter


def create_inpaint_mask(img):
    # threshold image
    thresh = 25
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(grey_img, thresh, 255, cv2.THRESH_BINARY)

    # find image contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # filter contours by area to remove salt n peppa noise contours
    min_area = 20
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    # missing region is 2nd largest contour
    inpainting_contour = contours[1]

    # initialise blank - black binary mask
    mask = np.zeros_like(grey_img)
    # draw contour on mask
    cv2.drawContours(mask, [inpainting_contour], 0, (255),
                     thickness=cv2.FILLED)

    # normalise mask to have values {0.0, 1.0}
    normalised_mask = mask / 255
    return normalised_mask


def fix_perspective(img):
    # threshold image
    thresh = 10
    greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binaryImg = cv2.threshold(greyImg, thresh, 255, cv2.THRESH_BINARY)

    # find image contours
    contours, _ = cv2.findContours(binaryImg, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # get corner points of largest contour
    contour = contours[0]
    topLeft = tuple(contour[contour[:, :, 0].argmin()][0])
    topRight = tuple(contour[contour[:, :, 1].argmin()][0])
    bottomLeft = tuple(contour[contour[:, :, 1].argmax()][0])
    bottomRight = tuple(contour[contour[:, :, 0].argmax()][0])

    # get perspective transform
    srcPoints = np.float32([topLeft, topRight, bottomLeft, bottomRight])
    dstPoints = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]],
                           [img.shape[1], img.shape[0]]])
    perspectiveMatrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)

    # apply perspective tansformation
    img = cv2.warpPerspective(img, perspectiveMatrix,
                              (img.shape[1], img.shape[0]))
    return img


def gamma_correction(img, gamma=1.5):
    # convert to lab colour space
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

    # normalise lumincance, apply gamma correction, de-normalise
    l_corrected = (np.power((l / 255), gamma) * 255).astype(np.uint8)

    # merge channels, convert to bgr
    return cv2.cvtColor(cv2.merge((l_corrected, a, b)), cv2.COLOR_LAB2BGR)


def apply_denoising(img):
    # using cv2 non-local means denoising
    return cv2.fastNlMeansDenoisingColored(img, None, 14, 8, 15, 19)


def get_file_list(path):
    try:
        file_names = os.listdir(path)
        return file_names
    except Exception as e:
        # any exception shoud cause program to terminate
        print(f"Exception occured accessing input files: {e}")
        sys.exit(1)


def create_folder(path):
    # check if folder already exists
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
    except Exception as e:
        # any exception shoud cause program to terminate
        print(f"Exception occured creating output folder: {e}")
        sys.exit(1)


def main(files, src_path, out_path):
    for file in files:
        print(file)
        # read image data
        img = cv2.imread(os.path.join(src_path, file), cv2.IMREAD_COLOR)
        if img is None:
            # unsuccessful image read => exit
            print(f"Error reading image data from file: {file}")
            sys.exit(1)

        # apply inpainting
        inpaint_mask = create_inpaint_mask(img)
        img = Inpainter(img, inpaint_mask, 12, 2.25).inpaint()

        # apply perspective transformation
        img = fix_perspective(img)

        # apply denoising
        img = cv2.fastNlMeansDenoisingColored(img, None, 14, 8, 15, 19)

        # enhance contrast with gamma correction
        img = gamma_correction(img, 1.4)

        # write image file to output
        cv2.imwrite(os.path.join(out_path, file), img)


# command-line input parser
parser = argparse.ArgumentParser(description="Process corrupted images")
parser.add_argument('path', type=str, help="path to image directory")
src_path = parser.parse_args().path
out_path = "./test_results"

# get file list
file_names = get_file_list(src_path)

# create output folder
create_folder(out_path)

# apply image processing
main(file_names, src_path, out_path)
