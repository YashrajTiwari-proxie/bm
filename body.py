'''
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# --- Step 1: Load image and segment person ---
image_path = "person.jpg"  # Replace with your own image file
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# MediaPipe segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
results = segmentor.process(image_rgb)
segmentation_mask = results.segmentation_mask > 0.5

# Create silhouette mask
silhouette_mask = np.zeros_like(image[:, :, 0])
silhouette_mask[segmentation_mask] = 255

# Optional cleanup with morphology
kernel = np.ones((5, 5), np.uint8)
silhouette_mask = cv2.morphologyEx(silhouette_mask, cv2.MORPH_CLOSE, kernel)

# Get top and bottom of silhouette
ys, _ = np.where(silhouette_mask == 255)
top_y, bottom_y = np.min(ys), np.max(ys)
height_px = bottom_y - top_y

# Get user input
real_height_cm = float(input("Enter person's real height in cm: "))
cm_per_pixel = real_height_cm / height_px

# Annotated image
annotated_mask = cv2.cvtColor(silhouette_mask, cv2.COLOR_GRAY2BGR)

# --- Step 2: Utility functions ---
def horizontal_width(mask, y):
    row = mask[y, :]
    x_indices = np.where(row == 255)[0]
    if len(x_indices) > 0:
        return x_indices[-1] - x_indices[0], x_indices[0], x_indices[-1]
    return 0, 0, 0

def find_dynamic_waist(mask):
    h, _ = mask.shape
    y_vals = []
    widths = []

    for y in range(int(0.4 * h), int(0.7 * h)):
        width, _, _ = horizontal_width(mask, y)
        y_vals.append(y)
        widths.append(width)

    widths = np.array(widths)
    y_vals = np.array(y_vals)
    waist_y = y_vals[np.argmin(widths)]
    return waist_y

def find_hip_after_waist(mask, waist_y):
    h, _ = mask.shape
    y_range = range(waist_y + int(0.05 * h), waist_y + int(0.20 * h))
    widths = []
    y_vals = []

    for y in y_range:
        width, _, _ = horizontal_width(mask, y)
        widths.append(width)
        y_vals.append(y)

    if not widths:
        return waist_y + 50

    widths = np.array(widths)
    y_vals = np.array(y_vals)
    hips_idx = np.argmax(widths)
    return y_vals[hips_idx]

def annotate_width(mask, y, label, annotated_img):
    width, x1, x2 = horizontal_width(mask, y)
    width_cm = width * cm_per_pixel
    cv2.line(annotated_img, (x1, y), (x2, y), (0, 255, 0), 2)
    cv2.putText(annotated_img, f"{label}: {width_cm:.1f} cm", (x1, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return width_cm

# --- Step 3: Compute and annotate landmarks ---
neck_y = int(top_y + 0.095 * height_px)
shoulder_y = int(top_y + 0.18 * height_px)
waist_y = find_dynamic_waist(silhouette_mask)
hip_y = find_hip_after_waist(silhouette_mask, waist_y)

neck_cm = annotate_width(silhouette_mask, neck_y, "neck", annotated_mask)
shoulder_cm = annotate_width(silhouette_mask, shoulder_y, "shoulders", annotated_mask)
waist_cm = annotate_width(silhouette_mask, waist_y, "waist", annotated_mask)
hip_cm = annotate_width(silhouette_mask, hip_y, "hips", annotated_mask)

# Torso length
torso_cm = (waist_y - neck_y) * cm_per_pixel
cv2.putText(annotated_mask, f"Torso: {torso_cm:.1f} cm", (10, bottom_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# --- Step 4: Show original and result side-by-side ---
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image_rgb)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(cv2.cvtColor(annotated_mask, cv2.COLOR_BGR2RGB))
axs[1].set_title("Silhouette with Measurements")
axs[1].axis("off")

plt.tight_layout()
plt.show()

# --- Step 5: Save output and print results ---
cv2.imwrite("generated_silhouette_mask.png", silhouette_mask)
cv2.imwrite("annotated_measurements.png", annotated_mask)

print("\nMeasurements (in cm):")
print(f"Neck: {neck_cm:.1f}")
print(f"Shoulders: {shoulder_cm:.1f}")
print(f"Waist: {waist_cm:.1f}")
print(f"Hips: {hip_cm:.1f}")
print(f"Torso Length: {torso_cm:.1f}")
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/MediaPipe logs

import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image
from io import BytesIO

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Body Measurement Estimator", page_icon="🧍‍♂️")

st.title("Body Measurement Estimator")
st.markdown("Upload a full-body image to estimate measurements.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and process the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show original image
    st.image(image_rgb, caption="Original Image", use_container_width=True)

    # MediaPipe segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentor:
        results = segmentor.process(image_rgb)
        segmentation_mask = results.segmentation_mask > 0.5

    # Create silhouette mask
    silhouette_mask = np.zeros_like(image[:, :, 0])
    silhouette_mask[segmentation_mask] = 255

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    silhouette_mask = cv2.morphologyEx(silhouette_mask, cv2.MORPH_CLOSE, kernel)

    # Find top and bottom of the silhouette
    ys, _ = np.where(silhouette_mask == 255)
    top_y, bottom_y = np.min(ys), np.max(ys)
    height_px = bottom_y - top_y

    # Get height from user
    real_height_cm = st.number_input("Enter person's real height in cm:", min_value=50.0, max_value=250.0, value=170.0)
    cm_per_pixel = real_height_cm / height_px
    annotated_mask = cv2.cvtColor(silhouette_mask, cv2.COLOR_GRAY2BGR)

    # --- Utility Functions ---
    def horizontal_width(mask, y):
        row = mask[y, :]
        x_indices = np.where(row == 255)[0]
        if len(x_indices) > 0:
            return x_indices[-1] - x_indices[0], x_indices[0], x_indices[-1]
        return 0, 0, 0

    def find_dynamic_waist(mask):
        h, _ = mask.shape
        y_vals = []
        widths = []

        # Adjust the waist detection to make sure it's closer to the correct position
        for y in range(int(0.4 * h), int(0.65 * h)):  # Adjust the range to focus on the upper body
            width, _, _ = horizontal_width(mask, y)
            y_vals.append(y)
            widths.append(width)

        widths = np.array(widths)
        y_vals = np.array(y_vals)
        waist_y = y_vals[np.argmin(widths)]
        return waist_y

    def find_hip_after_waist(mask, waist_y):
        h, _ = mask.shape
        y_range = range(waist_y + int(0.05 * h), waist_y + int(0.15 * h))  # Adjust hip range closer to the waist
        widths = []
        y_vals = []

        for y in y_range:
            width, _, _ = horizontal_width(mask, y)
            widths.append(width)
            y_vals.append(y)

        if not widths:
            return waist_y + 50

        widths = np.array(widths)
        y_vals = np.array(y_vals)
        hips_idx = np.argmax(widths)
        return y_vals[hips_idx]

    def annotate_width(mask, y, label, annotated_img):
        width, x1, x2 = horizontal_width(mask, y)
        width_cm = width * cm_per_pixel
        cv2.line(annotated_img, (x1, y), (x2, y), (0, 255, 0), 2)
        cv2.putText(annotated_img, f"{label}: {width_cm:.1f} cm", (x1, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return width_cm

    # --- Measurement Computation ---
    neck_y = int(top_y + 0.095 * height_px)
    shoulder_y = int(top_y + 0.18 * height_px)
    waist_y = find_dynamic_waist(silhouette_mask)  # Adjusted waist position
    hip_y = find_hip_after_waist(silhouette_mask, waist_y)  # Adjusted hip position

    neck_cm = annotate_width(silhouette_mask, neck_y, "Neck", annotated_mask)
    shoulder_cm = annotate_width(silhouette_mask, shoulder_y, "Shoulders", annotated_mask)
    waist_cm = annotate_width(silhouette_mask, waist_y, "Waist", annotated_mask)
    hip_cm = annotate_width(silhouette_mask, hip_y, "Hips", annotated_mask)

    torso_cm = (waist_y - neck_y) * cm_per_pixel
    cv2.putText(annotated_mask, f"Torso: {torso_cm:.1f} cm", (10, bottom_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- Show Annotated Image ---
    st.image(cv2.cvtColor(annotated_mask, cv2.COLOR_BGR2RGB),
             caption="Silhouette with Measurements", use_container_width=True)

    # --- Output Measurements ---
    st.subheader("Estimated Measurements")
    st.write(f"**Neck:** {neck_cm:.1f} cm")
    st.write(f"**Shoulders:** {shoulder_cm:.1f} cm")
    st.write(f"**Waist:** {waist_cm:.1f} cm")
    st.write(f"**Hips:** {hip_cm:.1f} cm")
    st.write(f"**Torso Length:** {torso_cm:.1f} cm")

    # --- Save Outputs ---
    cv2.imwrite("generated_silhouette_mask.png", silhouette_mask)
    cv2.imwrite("annotated_measurements.png", annotated_mask)
