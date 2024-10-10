# https://github.com/RapidAI/RapidStructure/blob/main/docs/README_Orientation.md
import cv2
from rapid_orientation import RapidOrientation

orientation_engine = RapidOrientation()

# img = cv2.imread("z_using_files/pics/11.jpg")
img = cv2.imread("upload_files/rotate_pics/11.jpg")
orientation_res, elapse = orientation_engine(img)
print("x" + orientation_res + "x")
