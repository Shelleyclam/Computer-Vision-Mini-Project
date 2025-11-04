print("\n--- UNIT 3: Hough Circle Detection ---")
gray = cv2.imread("img_circles.png", cv2.IMREAD_GRAYSCALE)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=20, minRadius=10, maxRadius=40)
out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
if circles is not None:
    for (x,y,r) in np.round(circles[0, :]).astype("int"):
        cv2.circle(out, (x, y), r, (0,255,0), 2)
        cv2.circle(out, (x, y), 2, (0,0,255), 3)
show(out, "Detected Circles")
