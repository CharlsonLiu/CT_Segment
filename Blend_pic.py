
import os
import cv2
import numpy as np

ct_folder_path = "DATA-3/test/JPEGImages"
label_folder_path = "unet-(pp)/outputs/DATA-3_UNet_woDS/0"
output_folder_path = "blend_pic/unet_test"

# Get the list of file paths for CT images in the folder
ct_file_paths = [f.path for f in os.scandir(ct_folder_path) if f.name.endswith('.jpg')]

# Sort the CT file paths by filename
ct_file_paths = sorted(ct_file_paths)

# Get the list of file paths for label images in the folder
label_file_paths = [f.path for f in os.scandir(label_folder_path) if f.name.endswith('.png')]

# Sort the label file paths by filename
label_file_paths = sorted(label_file_paths)

# Create the output folder if it doesn't already exist
os.makedirs(output_folder_path, exist_ok=True)

# Iterate through the file paths and load the corresponding images
for ct_path, label_path in zip(ct_file_paths, label_file_paths):
    # Load the CT image
    ct_img = cv2.imread(ct_path)

    # Load the label image
    label_img = cv2.imread(label_path)

    # Check the color mode of the input images
    print(ct_img.shape, label_img.shape)

    # Resize the label image to match the CT image size
    label_img = cv2.resize(label_img, ct_img.shape[:2][::-1])

    # Create a green array with the same shape as label_img
    green_arr = np.zeros_like(label_img)
    green_arr[:] = (0, 255, 0)

    # Replace white pixels with green
    mask = (label_img == (255, 255, 255)).all(axis=2)
    label_img[mask] = green_arr[mask]

    # Blend the label image with the CT image
    blended_img = cv2.addWeighted(ct_img, 0.5, label_img, 0.5, 0)

    # Save the blended image to the output folder
    filename = os.path.splitext(os.path.basename(ct_path))[0]
    output_path = os.path.join(output_folder_path, filename + ".jpg")
    cv2.imwrite(output_path, blended_img)