print("\n--- UNIT 1: Edge Detection + Thresholding ---")
img = cv2.imread("img_shapes.png", cv2.IMREAD_GRAYSCALE)
# Otsu threshold
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Canny edges
edges = cv2.Canny(img, 50, 150)
show(img, "Original")
show(otsu, "Otsu Thresholding")
show(edges, "Canny Edges")
