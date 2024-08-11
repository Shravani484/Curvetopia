import numpy as np
import matplotlib.pyplot as plt
import csv

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot_paths(paths_XYs, save_path='output_plot.png'):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = 'C' + str(i % 10)  # Cycle through color palette
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)

    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    # Remove axes
    ax.axis('off')

    # Save the plot as a PNG file without axes
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Display the plot (optional)
    plt.show()

# Load and visualize the polylines
csv_path = '/content/isolated.csv'  # Update with actual CSV file path
paths_XYs = read_csv(csv_path)
save_path='output_plot2.png'
# Plot and save the image without axes
plot_paths(paths_XYs, save_path)


import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread(save_path)

image_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

edges=cv2.Canny(image=image_rgb,threshold1=100,threshold2=700)

fig,axs=plt.subplots(1,2,figsize=(7,4))
axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')

axs[1].imshow(edges)
axs[1].set_title('Image Edges')

for ax in axs:
  ax.set_xticks([])
  ax.set_yticks([])

plt.tight_layout()
plt.show()


# Step 2: Apply Gaussian Blur to smooth the image (reduce noise)
blurred_image = cv2.GaussianBlur(image_rgb, (5, 5), 0)

# Step 4: Apply morphological operations to clean up the edges
# Dilation followed by erosion (closing) to remove small gaps
kernel = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)
edges_cleaned = cv2.erode(edges_dilated, kernel, iterations=1)

# Step 5: Find contours
contours, _ = cv2.findContours(edges_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Approximate contours with a smaller epsilon for more detail
approximated_contours = []
for cnt in contours:
    epsilon = 0.0125 * cv2.arcLength(cnt, True)  # Smaller epsilon for higher detail
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approximated_contours.append(approx)

# Step 7: Draw the approximated contours on a blank canvas
canvas = np.zeros_like(edges)
cv2.drawContours(canvas, approximated_contours, -1, (255, 255, 255), thickness=2)

# Step 8: Display the results
fig, axs = plt.subplots(1, 3, figsize=(12, 6))

axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')

axs[1].imshow(edges_cleaned, cmap='gray')
axs[1].set_title('Cleaned Edges')

axs[2].imshow(canvas, cmap='gray')
axs[2].set_title('Approximated Contours')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
