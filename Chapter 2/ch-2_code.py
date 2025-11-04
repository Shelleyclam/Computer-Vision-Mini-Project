print("\n--- UNIT 2: Connected Components + Region Descriptors ---")
labeled = measure.label(otsu, connectivity=2)
props = measure.regionprops(labeled)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for i, p in enumerate(props, start=1):
    y0, x0, y1, x1 = p.bbox
    cy, cx = p.centroid
    cv2.rectangle(img_color, (x0,y0), (x1,y1), (0,255,0), 1)
    cv2.circle(img_color, (int(cx), int(cy)), 3, (0,0,255), -1)
    cv2.putText(img_color, f"{i}", (x0, y0-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

print("Detected Objects:", len(props))
show(img_color, "Labeled Objects with Centroids")
