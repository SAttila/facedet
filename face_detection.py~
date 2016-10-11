import cv2
import numpy as np
import sys
import glob
import argparse
import skindetector as sd


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_nose_nariz.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mouth.xml')
profileface_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')

parser = argparse.ArgumentParser(description='Face detection - Attila Schulc')
parser.add_argument("-i", "--image", required=False, help="Image input or image sequence. Example: 'home/images/img_001.png' or 'home/images/*png'")
parser.add_argument("-v", "--video", required=False, help="Video input. Example: 'video_file.mp4'")
parser.add_argument("-c", "--camera", required=False, help="Camera input. Example: '0' - primary cam, '1' - secondary cam, ...")

args = vars(parser.parse_args())


# Function to eliminate false positives based on area ratio
def are_based_elimination(boxes, face_area, min_val, max_val):
    if len(boxes) == 0:
        return []

    indexes = []
    for i in range(0, len(boxes)):
        (x, y, w, h) = boxes[i]
        if w*h < face_area * min_val or face_area * max_val < w*h:
            indexes.append(i)

    return np.delete(boxes, indexes, axis=0)


def detect_eyes(roi_color, roi_gray, face_area):
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 5)
    eyes = are_based_elimination(eyes, face_area, 0.01, 0.06)

    for (x, y, w, h) in eyes:
        cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(roi_color, (x+int(w/2), y+int(h/2)), 2, (0, 255, 0), 1)


def detect_nose(roi_color, roi_gray, face_area):
    nose = nose_cascade.detectMultiScale(roi_gray, 1.05, 5)
    nose = are_based_elimination(nose, face_area, 0.01, 0.06)
    for (x, y, w, h) in nose:
        cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 200, 200), 2)
        cv2.circle(roi_color, (x+int(w/2), y+int(h/2)), 2, (0, 200, 200), 1)


def detect_mouth(roi_color, roi_gray, face_area):
    mouth = mouth_cascade.detectMultiScale(roi_gray)
    mouth = are_based_elimination(mouth, face_area, 0.01, 0.06)

    # mouth = non_max_suppression_slow(mouth, 0.3)

    for (x, y, w, h) in mouth:
        cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(roi_color, (x + int(w / 2), y + int(h / 2)), 2, (0, 0, 255), 1)


def detect_face(frame):
    try:
        # h, w, _ = frame.shape
        # new_width = 640
        # frame = cv2.resize(frame, (new_width, int(h / (w / new_width))), 0, 0)

        # My silly solution for face detection based on skin color
        # Assumes one face and good lighting conditions
        skin_mask = sd.detect_skin(frame)
        skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
        pixel_points = cv2.findNonZero(skin_mask)
        bx, by, bw, bh = cv2.boundingRect(pixel_points)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)
        if len(faces) == 0:
            faces = profileface_cascade.detectMultiScale(gray)
            if len(faces) != 0:
                print('Profile face found')

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, (x+int(w/2), y+int(h/2)), 2, (255, 0, 0), 1)
            # Face region
            roi_face_gray = gray[y:y+h, x:x+w]
            roi_face_color = frame[y:y+h, x:x+w]

            # Eye region
            eye_top = int(h*0.2)
            eye_bottom = int(h*0.5)
            eye_left = int(w*0.1)
            eye_right = int(w*0.9)
            roi_subf_color = roi_face_color[eye_top:eye_bottom, eye_left:eye_right]
            roi_subf_gray = roi_face_gray[eye_top:eye_bottom, eye_left:eye_right]
            detect_eyes(roi_subf_color, roi_subf_gray, w*h)
            cv2.imshow("roi eyes", roi_subf_gray)

            # Nose region
            nose_top = int(h * 0.5)
            nose_bottom = int(h * 0.75)
            nose_left = int(w * 0.3)
            nose_right = int(w * 0.7)
            roi_subf_color = roi_face_color[nose_top:nose_bottom, nose_left:nose_right]
            roi_subf_gray = roi_face_gray[nose_top:nose_bottom, nose_left:nose_right]
            detect_nose(roi_subf_color, roi_subf_gray, w*h)
            cv2.imshow("roi nose", roi_subf_gray)

            # Mouth region
            mouth_top = int(h*0.75)
            mouth_bottom = h
            mouth_left = int(w*0.3)
            mouth_right = int(w*0.7)
            roi_subf_color = roi_face_color[mouth_top:mouth_bottom, mouth_left:mouth_right]
            roi_subf_gray = roi_face_gray[mouth_top:mouth_bottom, mouth_left:mouth_right]
            detect_mouth(roi_subf_color, roi_subf_gray, w*h)
            cv2.imshow("roi mouth", roi_subf_gray)

            skin_rect = sd.Rectangle(bx, by, bx+bw, by+bh)
            haar_face_rect = sd.Rectangle(x, y, x+w, y+h)
            haar_area = w*h

            if skin_rect != (0, 0, 0, 0):
                if sd.intersect_area(skin_rect, haar_face_rect)/haar_area > 0.5:
                    cv2.putText(frame, "Positive", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), thickness=2)

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 1)
        cv2.putText(frame, "skin", (bx + bw, by + bh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), thickness=1)
        cv2.imshow('skin', skin)
        cv2.imshow("Frame", frame)
    except:
        e = sys.exc_info()
        print(e)


parser.print_help()


def image_mode(x):
    images = glob.glob(x)
    for img in images:
        frame = cv2.imread(img)
        detect_face(frame)
        k = cv2.waitKey(0) & 0xff
        if k == 27 or k == ord('q'):
            break


def video_mode(x):
    cap = cv2.VideoCapture(x)
    while True:
        ret, frame = cap.read()
        detect_face(frame)
        k = cv2.waitKey(25) & 0xff
        if k == 27 or k == ord('q'):
            break

    cap.release()


if args['image']:
    image_mode(args['image'])

elif args['video']:
    video_mode(args['video'])

elif args['camera']:
    video_mode(int(args['camera']))

cv2.destroyAllWindows()

if __name__ == "__main__":
    # video_mode(0)
    # video_mode('capture.avi')
    cv2.destroyAllWindows()
