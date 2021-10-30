import cv2
import numpy as np

img = cv2.imread("img/tenge.jpg")

out_img = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 1.5)


def detect_circles(input_img):
    circles = cv2.HoughCircles(input_img, cv2.HOUGH_GRADIENT, 1, 175, param1=50, param2=30, minRadius=30, maxRadius=110)
    return np.uint16(np.around(circles))


detected_circles = detect_circles(gray)[0, :]

max_radius = 0
for (x, y, r) in detected_circles:
    if r > max_radius:
        max_radius = r

for (x, y, r) in detected_circles:
    if r == max_radius:
        cv2.circle(out_img, (x, y), r, (0, 255, 0), 3)
        cv2.circle(out_img, (x, y), 2, (0, 255, 0), 5)
        cv2.putText(out_img, "100 tenge", (x - 80, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                    thickness=2)

cv2.imshow("100 tenge", out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
