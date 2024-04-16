#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 2023

@author: wuyiqi
"""

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import json
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Load coco_files
with open('/Path/to/coco_annotation_json_file/from/R&G models/', 'r') as file:
    coco_data = json.load(file)

coco = COCO()
coco.dataset = coco_data
coco.createIndex()

# Initialize segmask for each image
decoded_masks = {}
for img in coco_data['images']:
    file_name = img['file_name']
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=img['id']))

    # Initialize an empty mask array
    height, width = img['height'], img['width']
    full_mask = np.zeros((height, width), dtype=bool)

    # Merge all masks of the current image
    for ann in annotations:
        if 'segmentation' in ann:
            segs = ann['segmentation']
            new_image = coco_mask.decode(segs) if isinstance(segs, list) else coco_mask.decode([segs])
            new_image = np.squeeze(new_image)  # Remove the extra dimension
            if new_image.ndim > 2:  # If new_image is a three-dimensional array, only the first channel is kept
                new_image = new_image[:, :, 0]
            full_mask = np.logical_or(full_mask, new_image)
        
    decoded_masks[file_name] = full_mask

# save mask
output_mask_dir = '/Path/to/output_mask_images/'
os.makedirs(output_mask_dir, exist_ok=True)

for file_name, mask in decoded_masks.items():
    mask_image = (mask * 255).astype('uint8')
    output_file = os.path.join(output_mask_dir, f'mask_{file_name}.png')
    cv2.imwrite(output_file, mask_image)

# Image to Binary matrix
def load_and_convert_to_binary(image_path):
    image = Image.open(image_path)
    binary_matrix = np.array(image.convert('L')) > 128  # 使用128作为阈值
    return binary_matrix.astype(int)

# jaccard_similarity
def jaccard_similarity_coefficient(matrix1, matrix2):
    intersection_size = np.sum(np.logical_and(matrix1, matrix2))
    jaccard_similarity = intersection_size / np.sum(np.logical_or(matrix1, matrix2))
    return jaccard_similarity

threshold = default #0.8
similarity_scores = {}
previous_mask = None
selected_frames = []

for img in coco_data['images']:
    file_name = img['file_name']
    current_mask = decoded_masks.get(file_name)
    
    if previous_mask is not None:
        jaccard_similarity = jaccard_similarity_coefficient(previous_mask, current_mask)
        similarity_scores[file_name] = jaccard_similarity
        if jaccard_similarity < threshold:
            selected_frames.append(file_name)
    previous_mask = current_mask

# 
print("Dissimilar Images:", [img['id'] for img in coco_data['images'] if img['file_name'] in selected_frames])
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.histplot(similarity_scores, kde=True, color='blue')

plt.title('Jaccard Similarity Coefficients Distribution')
plt.xlabel('Jaccard Similarity Coefficient')
plt.ylabel('Frequency')
plt.savefig('output/**.png', format='png', dpi=1200)
plt.show()

# new coco_file
filtered_coco_data = {
    'images': [img for img in coco_data['images'] if img['file_name'] in selected_frames],
    'annotations': [anno for anno in coco_data['annotations'] if anno['image_id'] in [img['id'] for img in coco_data['images'] if img['file_name'] in selected_frames]],
    'categories': coco_data['categories']
}

with open('/Path/to/new_filtered_coco_file', 'w') as file:
    json.dump(filtered_coco_data, file, indent=4)

with open('/Path/to/new_filtered_coco_file', 'r') as file:
    filtered_coco_data = json.load(file)

# extract frames 
frame_indices = [int(img['file_name'].split('_')[-1]) for img in filtered_coco_data['images']]

video_path = '/Path/to/your/video_file/'
cap = cv2.VideoCapture(video_path)

output_dir = '/Path/to/selected_frames/'
os.makedirs(output_dir, exist_ok=True)

# Video process
Frame_count = 0
success, frame = cap.read()
while success:
    if Frame_count in frame_indices:
        
        frame_filename = os.path.join(output_dir, f'frame_{Frame_count}.jpg')
        cv2.imwrite(frame_filename, frame)
    
    success, frame = cap.read()
    Frame_count += 1

cap.release()
