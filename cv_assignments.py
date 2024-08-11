import cv2
import numpy as np

# SIFT feature detector
def SIFT(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    image = cv2.drawKeypoints(gray, keypoints, image)
    return keypoints, descriptors



# Binary descriptor
def ORB(image):
    orb = cv2.ORB_create()  #initial ORB object
    keypoints = orb.detect(image,None)   # detect the keypoints (mask = none)
    keypoints, descriptors = orb.compute(image,keypoints)  # calculate the decriptors by keypoints

    image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)  # green
    return keypoints, descriptors, image



# used to return the keypoints and descriptors but not draw these on image
def SIFT_with_no_kp_on_iamges(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


# Harris Corner feature detection
def harris_corner_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # turn to grayscale
    gray = np.float32(gray)  # turn to 32 float format
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)  # corner detect
    dst = cv2.dilate(dst, None)  # Expansion results
    img[dst > 0.01 * dst.max()] = [0, 0, 255]  # mark the corner as red

    # Extract corner locations as key points
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [cv2.KeyPoint(x=float(pt[1]), y=float(pt[0]), size = 10) for pt in keypoints]
    return keypoints, None, img


#SSD with ratio
def match_features_ssd_and_ratio(descriptors1, descriptors2):

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2) #return 2 nearest results

    good_matches = []
    for m, n in matches:  # calculate the squre of each distance

        ssd_m = m.distance ** 2
        ssd_n = n.distance ** 2

        if ssd_m < 0.75 ** 2 * ssd_n:  # test
            good_matches.append(m)

    return good_matches


# regular SSD, take into descriptors used to match
def match_features_ssd(descriptors1, descriptors2):
    #
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # initial a matcher with L2 Norm
    matches = matcher.match(descriptors1, descriptors2)

    #square the distance to follow SSD
    for m in matches:
        m.distance = m.distance ** 2

    #  k = 1, so there is only one match per matching group
    good_matches = sorted(matches, key=lambda x: x.distance) #sort the SSD and return the sorted matches list

    return good_matches



def resize_image_to_fit_screen(image, screen_width, screen_height):
    height, width = image.shape[:2]

    # calculate the ratio
    width_ratio = screen_width / width
    height_ratio = screen_height / height

    # maintain the aspect ratio
    ratio = min(width_ratio, height_ratio)

    # calculate new height and width
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Zoom image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image



# Image Stitching
def stitch_images(image1, image2, keypoints1, keypoints2, matches):
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]) #Extract coordinates of matching points
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    #Calculate the monoclinicity matrix
    H, status = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    width = image1.shape[1] + image2.shape[1] # calculate the stitch image width
    result = cv2.warpPerspective(image2, H, (width, image1.shape[0])) #Align the second image with the first
    result[0:image1.shape[0], 0:image1.shape[1]] = image1
    result = resize_image_to_fit_screen(result,1920,1080)  # resize the image that can all shown on screen
    return result



