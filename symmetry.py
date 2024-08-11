import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image and convert it to grayscale
image = cv2.imread('output_plot2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Detect edges using Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Step 4: Find contours with hierarchy to detect nested contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Filter and approximate the contours to retain both rectangles and circles
rectified_contours = []
for i, contour in enumerate(contours):
    epsilon = cv2.arcLength(contour, True) * 0.02
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Calculate the bounding box and aspect ratio
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h

    # Retain the contour if it's likely a rectangle (aspect ratio close to 1) or circle (high number of points)
    if len(approx) >= 4 or (0.9 <= aspect_ratio <= 1.1):
        rectified_contours.append(approx)

# Step 6: Draw the rectified contours on a new canvas
rectified_canvas = np.zeros_like(image)
cv2.drawContours(rectified_canvas, rectified_contours, -1, (255, 255, 255), thickness=2)

# Step 7: Convert the rectified canvas to RGB for display
rectified_canvas_rgb = cv2.cvtColor(rectified_canvas, cv2.COLOR_BGR2RGB)

# Step 8: Display the original image and the rectified result side by side using Matplotlib
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

axs[1].imshow(rectified_canvas_rgb)
axs[1].set_title('Rectified Shapes')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# Step 9: Further processing of rectified_canvas_rgb to find symmetry
gray = cv2.cvtColor(rectified_canvas_rgb, cv2.COLOR_RGB2GRAY)

# Step 10: Create horizontally and vertically flipped versions
flip_horizontal = cv2.flip(gray, 0)
flip_vertical = cv2.flip(gray, 1)

# Step 11: Draw symmetry lines on the rectified canvas
img_symmetry = cv2.cvtColor(rectified_canvas_rgb, cv2.COLOR_RGB2BGR).copy()
cv2.line(img_symmetry, (0, gray.shape[0]//2), (gray.shape[1], gray.shape[0]//2), (0, 0, 255), 2)  # Horizontal line
cv2.line(img_symmetry, (gray.shape[1]//2, 0), (gray.shape[1]//2, gray.shape[0]), (0, 255, 0), 2)  # Vertical line

# Step 12: Display the final image with symmetry lines
cv2.imshow('Symmetry Lines', img_symmetry)
cv2.waitKey(0)

# Step 13: Analyze and visualize individual shape symmetry (both horizontal and vertical)
for i, contour in enumerate(rectified_contours):
    # Create a mask for the contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Extract the region of interest (ROI) and the corresponding mask
    x, y, w, h = cv2.boundingRect(contour)
    roi = mask[y:y+h, x:x+w]

    # Flip the ROI horizontally and vertically
    flipped_roi_h = cv2.flip(roi, 0)
    flipped_roi_v = cv2.flip(roi, 1)

    # Create new images to display the original and flipped ROIs side by side
    combined_roi_h = np.vstack((roi, flipped_roi_h))
    combined_roi_v = np.hstack((roi, flipped_roi_v))

    # Draw lines indicating the axes of symmetry
    cv2.line(combined_roi_h, (0, h), (w, h), (255, 0, 0), 1)  # Horizontal axis
    cv2.line(combined_roi_v, (w, 0), (w, h), (255, 0, 0), 1)  # Vertical axis

    # Display the results for each shape
    cv2.imshow(f'Horizontal Symmetry {i}', combined_roi_h)
    cv2.waitKey(0)
    cv2.imshow(f'Vertical Symmetry {i}', combined_roi_v)
    cv2.waitKey(0)

# Step 14: Display the final rectified shapes
cv2.imshow('Rectified Shapes', rectified_canvas_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
