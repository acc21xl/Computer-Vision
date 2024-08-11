import cv2
import tkinter as tk
from tkinter import Toplevel
from PIL import Image, ImageTk
import time

from cv_assignments import SIFT, match_features_ssd_and_ratio, stitch_images, resize_image_to_fit_screen, harris_corner_detection, SIFT_with_no_kp_on_iamges, ORB


def setup_ui(window):
    frame = tk.Frame(window)
    frame.pack(fill=tk.BOTH, expand=True)



    #The data is from 1 to 5 with both left and right
    #copy the below path to run different images
    # 1: r"dataset\data1_left.jpg", r"dataset\data1_right.jpg"
    # 2: r"dataset\data2_left.jpg", r"dataset\data2_right.jpg"
    # 3: r"dataset\data3_left.jpg", r"dataset\data3_right.jpg"
    # 4: r"dataset\data4_left.jpg", r"dataset\data4_right.jpg"
    # 5: r"dataset\data5_left.png", r"dataset\data5_right.png"

    SIFT_image = cv2.imread(r"dataset\data2_left.jpg")
    Harris_image = cv2.imread(r"dataset\data2_right.jpg")
    ORB_image = cv2.imread(r"dataset\data5_left.png")
    matching_image_left = cv2.imread(r"dataset\data5_left.png")
    matching_image_right = cv2.imread(r"dataset\data5_right.png")
    stitching_image_left = cv2.imread(r"dataset\data5_left.png")
    stitching_image_right = cv2.imread(r"dataset\data5_right.png")


    # Button for displaying feature detection results
    btn_SIFT_feature_detection = tk.Button(frame, text="Show SIFT Feature Detection", command=lambda: show_feature_detection(window, SIFT_image))
    btn_SIFT_feature_detection.pack(fill=tk.X)

    btn_Harris_feature_detection = tk.Button(frame, text="Show Harris Corner Feature Detection", command=lambda: show_harris_feature_detection(window, Harris_image))
    btn_Harris_feature_detection.pack(fill=tk.X)

    btn_ORB_descriptor = tk.Button(frame, text="Show ORB descripter", command=lambda: show_ORB_detection(window, ORB_image))
    btn_ORB_descriptor.pack(fill=tk.X)

    # button for displaying feature matching results
    btn_feature_matching = tk.Button(frame, text="Show Feature Matching", command=lambda: show_feature_matching(window, matching_image_left, matching_image_right))
    btn_feature_matching.pack(fill=tk.X)

    # button for displaying the image stitching results
    btn_image_stitching = tk.Button(frame, text="Show Image Stitching", command=lambda: show_image_stitching(window, stitching_image_left, stitching_image_right))
    btn_image_stitching.pack(fill=tk.X)


    return frame

def clear_display(window):
    # Clear or close only the top-level window, keeping the main window
    for widget in window.winfo_children():
        if isinstance(widget, Toplevel):
            widget.destroy()




#------------------------------------------------------------Harris Corner-----------------------------------------------------------------------------
def display_Harris_keypoints(top, image, keypoints, title="Harris Features"):
    # Converting images to formats available in Tkinter
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image_pil)

    for widget in top.winfo_children():
        widget.destroy()

    # Creating a canvas to display an image
    canvas = tk.Canvas(top, width=image_pil.width, height=image_pil.height)
    canvas.pack(side="left")
    canvas.create_image(0, 0, anchor="nw", image=image_tk)
    canvas.image = image_tk  # prevent image been missing

    # build the data line with coordinates and orientation
    text = tk.Text(top, height=20, width=60)
    text.pack(side="right", fill="both", expand=True)
    text.insert(tk.END, "Keypoints Information:\n")
    for i, kp in enumerate(keypoints):
        text.insert(tk.END, f"Keypoint {i+1}: X={kp.pt[0]:.2f}, Y={kp.pt[1]:.2f}, Orientation={kp.angle:.2f}°\n")

    top.title(title)


#show the harris result
def show_harris_feature_detection(window, image1):

    keypoints1, _, _ = harris_corner_detection(image1)
    clear_display(window)
    top = Toplevel(window)
    top.title("Harris Detection Results")
    tk.Label(top, text="Harris Detection Results Here").pack()
    image1 = resize_image_to_fit_screen(image1, 1400, 600)
    display_Harris_keypoints(top, image1, keypoints1, "Harris Corner Feature Detection")





#---------------------------------------------------------------SIFT-----------------------------------------------------------------------------


def display_SIFT_keypoints(top, image, keypoints, title="SIFT Features"):

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image_pil)

    for widget in top.winfo_children():
        widget.destroy()


    canvas = tk.Canvas(top, width=image_pil.width, height=image_pil.height)
    canvas.pack(side="left")
    canvas.create_image(0, 0, anchor="nw", image=image_tk)
    canvas.image = image_tk


    text = tk.Text(top, height=20, width=60)
    text.pack(side="right", fill="both", expand=True)
    text.insert(tk.END, "Keypoints Information:\n")
    for i, kp in enumerate(keypoints):
        text.insert(tk.END, f"Keypoint {i+1}: X={kp.pt[0]:.2f}, Y={kp.pt[1]:.2f}, Orientation={kp.angle:.2f}°\n")

    top.title(title)


def show_feature_detection(window, image1):
    #run the SIFT here to collect keypoints and show
    keypoints, _ = SIFT(image1)
    clear_display(window)
    top = Toplevel(window)
    top.title("SIFT Detection Results")
    tk.Label(top, text="SIFT Detection Results").pack()
    image1 = resize_image_to_fit_screen(image1, 1400, 600)
    display_SIFT_keypoints(top, image1, keypoints, "SIFT Feature Detection")


#--------------------------------------------------------ORB--------------------------------------------------------------

def display_ORB_keypoints(top, image, keypoints, title = "ORB Feature"):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image_pil)

    for widget in top.winfo_children():
        widget.destroy()

    canvas = tk.Canvas(top, width=image_pil.width, height=image_pil.height)
    canvas.pack(side="left")
    canvas.create_image(0, 0, anchor="nw", image=image_tk)
    canvas.image = image_tk

    top.title(title)

def show_ORB_detection(window, image1):
    keypoints, descriptors, orb_image = ORB(image1)
    clear_display(window)
    top = Toplevel(window)
    top.title("ORB Detection Results")
    tk.Label(top, text="ORB Detection Results").pack()
    image1 = resize_image_to_fit_screen(image1, 1400, 600)
    display_ORB_keypoints(top, orb_image, keypoints, "ORB Feature")

#--------------------------------------------------------matching--------------------------------------------------------------

def display_matches(top, img1, keypoints1, img2, keypoints2, matches, title="Feature Matching"):
    # Plotting Matching Results
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    matched_image = resize_image_to_fit_screen(matched_image,1200,720)
    cv2.imwrite(r"C:\Users\ALIENWARE\Desktop\third_year\second_semester\Computer_Vision\assignment\match_images\match_image.png",matched_image)


    image_pil = Image.fromarray(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image_pil)


    for widget in top.winfo_children():
        widget.destroy()

    canvas = tk.Canvas(top, width=image_pil.width, height=image_pil.height)
    canvas.pack(side="top", fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=image_tk)
    canvas.image = image_tk

    top.title(title)


def show_feature_matching(window, image1, image2):
    start_time = time.time()
    # SIFT or ORB, uncomment ORB if using ORB
    # keypoints1, descriptors1, orb_image = ORB(image1)
    # keypoints2, descriptors2, orb_image = ORB(image2)
    keypoints1, descriptors1 = SIFT_with_no_kp_on_iamges(image1)
    keypoints2, descriptors2 = SIFT_with_no_kp_on_iamges(image2)

    # SSD with ratio or match_features_ssd
    # matches = match_features_ssd(descriptors1, descriptors2)
    matches = match_features_ssd_and_ratio(descriptors1, descriptors2)

    clear_display(window)
    top = Toplevel(window)
    top.title("Feature Matching Results")
    tk.Label(top, text="Feature Matching Results").pack()

    display_matches(top, image1, keypoints1, image2, keypoints2, matches)
    end_time = time.time()
    print("Matching Time = ", end_time-start_time)

#----------------------------------------------------------Stitch Images--------------------------------------------------------------------------

def display_stitched_image(top, image, title="Stitched Image"):

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tk = ImageTk.PhotoImage(image_pil)


    for widget in top.winfo_children():
        widget.destroy()


    canvas = tk.Canvas(top, width=image_pil.width, height=image_pil.height)
    canvas.pack(side="top", fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=image_tk)
    canvas.image = image_tk

    top.title(title)


def show_image_stitching(window, image1, image2):


    start_time = time.time()
    keypoints1, descriptors1 = SIFT_with_no_kp_on_iamges(image1)
    keypoints2, descriptors2 = SIFT_with_no_kp_on_iamges(image2)

    #uncomment if using ORB
    # keypoints1, descriptors1, orb_image = ORB(image1)
    # keypoints2, descriptors2, orb_image = ORB(image2)

    #uncomment if using normal ssd
    # matches = match_features_ssd(descriptors1, descriptors2)
    matches = match_features_ssd_and_ratio(descriptors1, descriptors2)

    stitched_image = stitch_images(image1, image2, keypoints1, keypoints2, matches)
    stitched_image = resize_image_to_fit_screen(stitched_image,1640,1080) # resize to the window width and height

    clear_display(window)
    top = Toplevel(window)
    top.title("Image Stitching Results")
    tk.Label(top, text="Image Stitching Results").pack()

    display_stitched_image(top, stitched_image)
    end_time = time.time()
    print("total time = ",end_time-start_time, "seconds")




def main():
    root = tk.Tk()
    setup_ui(root)
    root.mainloop()

if __name__ == '__main__':
    main()


