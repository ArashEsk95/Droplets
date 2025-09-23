import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

# Parameters
video_path = "12K.mp4"
frame_limit = 10 #100000
distance_thresh = 3
max_lost = 6        # How many frames to keep a missing object
min_lifetime = 6    # Minimum frames a droplet must exist before drawing
resize_factor = 0.5  # 🔹 Resize video window (0.5 = half size)

# Tracking storage
next_id = 0
objects = {}  # id: {'center': (x, y), 'lost': 0, 'lifetime': 1}
trajectories = defaultdict(list)  # id: [list of centers]
all_radii = []  # List of detected droplet sizes (radii in pixels)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Cannot open video file!")
    exit()

frame_index = 0

while cap.isOpened() and frame_index < frame_limit:
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold to detect dark droplets on light background
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Filter contours by area and additional criteria to remove false positives
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    detected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if 20 < area < 250 and perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.4 < circularity <= 1.0:  # Circularity close to 1 indicates a circle
                filtered_contours.append(cnt)
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                detected.append((center, radius))
                all_radii.append(radius)  # Store for size distribution

    # updated_ids = set()
    # for det in detected:
    #     center = det[0]
    #     matched_id = None
    #     min_dist = float('inf')
    #     for obj_id, obj in objects.items():
    #         dist = np.linalg.norm(np.array(center) - np.array(obj['center']))
    #         if dist < distance_thresh and dist < min_dist:
    #             matched_id = obj_id
    #             min_dist = dist
    #     if matched_id is not None:
    #         objects[matched_id]['center'] = center
    #         objects[matched_id]['lost'] = 0
    #         objects[matched_id]['lifetime'] += 1
    #         trajectories[matched_id].append(center)
    #         updated_ids.add(matched_id)
    #     else:
    #         objects[next_id] = {'center': center, 'lost': 0, 'lifetime': 1}
    #         trajectories[next_id].append(center)
    #         updated_ids.add(next_id)
    #         next_id += 1

    # for obj_id in list(objects.keys()):
    #     if obj_id not in updated_ids:
    #         objects[obj_id]['lost'] += 1
    #         if objects[obj_id]['lost'] > max_lost:
    #             del objects[obj_id]

    # Draw stable droplets
    output = frame.copy()
    # Draw the actual contours on the output frame
    # cv2.drawContours(output, filtered_contours, -1, (0, 255, 0), 1)
    for center, radius in detected:
        # Draw a circle around the contour
        cv2.circle(output, center, int(radius), (0, 255, 0), 1)  # Blue circle
        # Draw a point at the center of the contour
        cv2.circle(output, center, 1, (0, 255, 0), -1)  # Red point

    # Save the frame with detected droplets
    output_filename = f"frame_{frame_index}.jpg"
    cv2.imwrite(output_filename, output)
    print(f"Saved: {output_filename}")

    # Resize and display
    output_resized = cv2.resize(output, None, fx=resize_factor, fy=resize_factor)
    cv2.imshow("Persistent Droplet Tracking", output_resized)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

    print(f"Frame {frame_index}, Active droplets: {len(objects)}")

cap.release()
cv2.destroyAllWindows()

# Plot Trajectories
# plt.figure(figsize=(10, 6))
# for traj in trajectories.values():
#     if len(traj) >= min_lifetime:
#         traj = np.array(traj)
#         plt.plot(traj[:, 0], traj[:, 1], marker='o', linewidth=1)
# plt.gca().invert_yaxis()
# plt.title("Droplet Trajectories")
# plt.xlabel("X Position (pixels)")
# plt.ylabel("Y Position (pixels)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Plot Droplet Size Distribution
# plt.figure(figsize=(8, 5))
# real_radius = all_radii * 20
# plt.hist(real_radius, bins=20, color='skyblue', edgecolor='black')
# plt.title("Droplet Size Distribution")
# plt.xlabel("Radius (Micron)")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
