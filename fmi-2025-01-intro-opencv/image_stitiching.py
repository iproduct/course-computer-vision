import cv2
import numpy as np

# 1. Load the two images to stitch
img1 = cv2.imread('data/mount_left.png')
img2 = cv2.imread('data/mount_right.png')

# 2. Initialize the feature detector (SIFT is excellent for stitching)
sift = cv2.SIFT_create()

# 3. Find keypoints (coordinates) and descriptors (unique visual fingerprints)
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# print(kp1, des1)
# print(kp2, des2)

# 4. Use a Flann-based matcher to find closest matching signatures
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
matches = matcher.knnMatch(des1, des2, k=2)

# 5. Apply Lowe's ratio test to filter out weak, ambiguous matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 6. Convert matching keypoints into arrays of (x, y) coordinates
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# print(src_pts.shape)
# print(src_pts)
# print(dst_pts.shape)

# 7. Compute the 3x3 Homography Matrix
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 8. Try inverting the homography matrix before warping
# H_inverse = np.linalg.inv(H)
# width = img1.shape[1] + img2.shape[1]
# height = max(img1.shape[0], img2.shape[0])
# result = cv2.warpPerspective(img1, H_inverse, (width, height))

# 7. Calculate boundaries to prevent coordinate clipping
print("Step 4: Creating translation space and dynamic canvas...")
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
warped_corners = cv2.perspectiveTransform(corners_img1, H)
corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

all_corners = np.concatenate((warped_corners, corners_img2), axis=0)
x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

# Create translation offsets to shift negative dimensions into viewable space
translation_dist = [-x_min, -y_min]
T = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

# Pre-multiply matrices to integrate shifting directly into the warp
H_shifted = T.dot(H)

# 8. Warp the first image plane onto the expanded dynamic canvas
output_width = x_max - x_min
output_height = y_max - y_min
result = cv2.warpPerspective(img1, H_shifted, (output_width, output_height))

# 9. Bitwise masking to erase sub-pixel rounding gap artifacts
print("Step 5: Mask-blending image borders to eliminate white seams...")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
_, binary_mask = cv2.threshold(img2_gray, 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(binary_mask)

# Isolate coordinates matching the shifted landing pad of img2
x_offset, y_offset = translation_dist[0], translation_dist[1]
roi = result[y_offset:y_offset + h2, x_offset:x_offset + w2]

# Carve a perfect geometric template hole inside the background canvas patch
roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
roi_fg = cv2.bitwise_and(img2, img2, mask=binary_mask)

# Flatten the image composite and drop it into place
dst = cv2.add(roi_bg, roi_fg)
result[y_offset:y_offset + h2, x_offset:x_offset + w2] = dst

# 10. Show the stitched result
try:
    cv2.imshow('Stitched Panorama Output', result)
    print("Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    pass
# cv2.imwrite('stitched_panorama.png', result)
