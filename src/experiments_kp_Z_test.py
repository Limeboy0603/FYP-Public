import cv2
from mp_util_legacy import mediapipe_extract_single, get_raw_coords, create_holistic
from math import isnan

# TESTING ONLY, DO NOT RUN

image_path = "dataset/full_test/frames/0.png"
image = cv2.imread(image_path)

# scale down the image, then pad the border back to 1920x1080
rows, cols, _ = image.shape
scale_factor = 0.8
resized_image = cv2.resize(image, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
# print(resized_image.shape)

top = (1080 - resized_image.shape[0]) // 2
bottom = 1080 - resized_image.shape[0] - top
left = (1920 - resized_image.shape[1]) // 2
right = 1920 - resized_image.shape[1] - left

resized_image = cv2.copyMakeBorder(
    resized_image,
    top,
    bottom,
    left,
    right,
    cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
)

holistic = create_holistic()
kp_original = mediapipe_extract_single(image, holistic)
_, face_original, left_hand_original, right_hand_original = get_raw_coords(kp_original)
del holistic

holistic = create_holistic()
kp_resized = mediapipe_extract_single(resized_image, holistic)
_, face_resized, left_hand_resized, right_hand_resized = get_raw_coords(kp_resized)
del holistic

face_len = len(face_original)
left_hand_len = len(left_hand_original)
right_hand_len = len(right_hand_original)

# for each (X, Y, Z) coordinate in both original and resized, print the Z coordinate difference

percent_diff_list = []

print("Face")
for i in range(face_len):
    original_z = face_original[i][2]
    resized_z = face_resized[i][2]
    diff = original_z - resized_z
    percent_diff = 100 * diff / original_z
    percent_diff_list.append(percent_diff) if not isnan(percent_diff) else None
    print(f"Original: {original_z}, Resized: {resized_z}, Diff: {diff}, Percent Diff: {percent_diff}")

print("Left Hand")
for i in range(left_hand_len):
    original_z = left_hand_original[i][2]
    resized_z = left_hand_resized[i][2]
    diff = original_z - resized_z
    percent_diff = 100 * diff / original_z
    percent_diff_list.append(percent_diff) if not isnan(percent_diff) else None
    print(f"Original: {original_z}, Resized: {resized_z}, Diff: {diff}, Percent Diff: {percent_diff}")

print("Right Hand")
for i in range(right_hand_len):
    original_z = right_hand_original[i][2]
    resized_z = right_hand_resized[i][2]
    diff = original_z - resized_z
    percent_diff = 100 * diff / original_z
    percent_diff_list.append(percent_diff) if not isnan(percent_diff) else None
    print(f"Original: {original_z}, Resized: {resized_z}, Diff: {diff}, Percent Diff: {percent_diff}")

# print average percent difference
print("Average Percent Diff:", sum(percent_diff_list) / len(percent_diff_list))