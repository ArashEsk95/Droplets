import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from scipy.optimize import linear_sum_assignment

# ------------------------- Parameters -------------------------
video_path_cam1 = "12K.mp4"
video_path_cam2 = "12K.mp4"
frame_limit = 100 #100000
angle_between_cameras_deg = 90
pixel_size_microns = 2
min_lifetime = 12
radius_similarity_thresh = 0.3  # 30% max radius difference allowed for match
position_thresh_pixels = 20     # Max distance between centers to be considered

angle_rad = math.radians(angle_between_cameras_deg)

# ------------------------- Tracking Function -------------------------
def track_droplets_with_radius(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return {}

    frame_index = 0
    distance_thresh = 3
    max_lost = 6
    next_id = 0
    objects = {}
    trajectories = defaultdict(list)

    while cap.isOpened() and frame_index < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 150:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                detected.append(((int(x), int(y)), radius))

        updated_ids = set()
        for center, radius in detected:
            matched_id = None
            min_dist = float('inf')
            for obj_id, obj in objects.items():
                dist = np.linalg.norm(np.array(center) - np.array(obj['center']))
                if dist < distance_thresh and dist < min_dist:
                    matched_id = obj_id
                    min_dist = dist
            if matched_id is not None:
                objects[matched_id]['center'] = center
                objects[matched_id]['radius'] = radius
                objects[matched_id]['lost'] = 0
                objects[matched_id]['lifetime'] += 1
                trajectories[matched_id].append((center, radius))
                updated_ids.add(matched_id)
            else:
                objects[next_id] = {'center': center, 'radius': radius, 'lost': 0, 'lifetime': 1}
                trajectories[next_id].append((center, radius))
                updated_ids.add(next_id)
                next_id += 1

        for obj_id in list(objects.keys()):
            if obj_id not in updated_ids:
                objects[obj_id]['lost'] += 1
                if objects[obj_id]['lost'] > max_lost:
                    del objects[obj_id]

    cap.release()
    return trajectories

# ------------------------- Load and Match -------------------------
print("Tracking droplets in both cameras...")
trajectories_1 = track_droplets_with_radius(video_path_cam1)
trajectories_2 = track_droplets_with_radius(video_path_cam2)

# ------------------------- Matching Using Hungarian -------------------------
print("Matching droplets...")
matched_pairs = []  # (id1, id2)
for id1, traj1 in trajectories_1.items():
    if len(traj1) < min_lifetime:
        continue
    last_pos1, r1 = traj1[-1]

    costs = []
    candidates = []
    for id2, traj2 in trajectories_2.items():
        if len(traj2) < min_lifetime:
            continue
        last_pos2, r2 = traj2[-1]

        # Check radius similarity
        radius_diff = abs(r1 - r2) / max(r1, r2)
        if radius_diff > radius_similarity_thresh:
            continue

        # Check distance threshold
        d = np.linalg.norm(np.array(last_pos1) - np.array(last_pos2))
        if d > position_thresh_pixels:
            continue

        costs.append(d)
        candidates.append(id2)

    if costs:
        min_idx = int(np.argmin(costs))
        matched_pairs.append((id1, candidates[min_idx]))

# ------------------------- Build 3D Trajectories -------------------------
trajectories_3d = []
for id1, id2 in matched_pairs:
    traj1 = trajectories_1[id1]
    traj2 = trajectories_2[id2]
    min_len = min(len(traj1), len(traj2))

    traj_3d = []
    for i in range(min_len):
        (x1, y1), _ = traj1[i]
        (z2, y2), _ = traj2[i]
        y_real = (y1 + y2) / 2

        x = x1 * pixel_size_microns
        y = y_real * pixel_size_microns
        z = z2 * pixel_size_microns / math.sin(angle_rad)
        traj_3d.append((x, y, z))
    trajectories_3d.append(np.array(traj_3d))

# ------------------------- Plot -------------------------
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for traj in trajectories_3d:
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=1)
ax.set_title("3D Droplet Trajectories (Hungarian Matching + Radius Check)")
ax.set_xlabel("X (μm)")
ax.set_ylabel("Y (μm)")
ax.set_zlabel("Z (μm)")
plt.tight_layout()
plt.show()
