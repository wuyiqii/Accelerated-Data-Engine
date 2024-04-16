#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 2023

@author: wuyiqi
"""

import json
import cv2
import os

# Load COCO JSON file
json_path = "/Path/to/coco_annotation_json_file/from/R&G models/"
with open(json_path, "r") as file:
    coco_data = json.load(file)

# count bbox number
bbox_counts = {}
for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    if image_id not in bbox_counts:
        bbox_counts[image_id] = 0
    bbox_counts[image_id] += 1

# count mean bbox number per frame
average_bbox_counts = {}
for i in range(len(coco_data["images"])):
    sum_prev = sum(bbox_counts.get(i - j, 0) for j in range(1, 4))  # i-3, i-2, i-1
    sum_next = sum(bbox_counts.get(i - j + 1, 0) for j in range(3))  # i-2, i-1, i
    average_prev = sum_prev / 3
    average_next = sum_next / 3
    average_bbox_counts[i] = (average_prev, average_next)

# filter frames
selected_frames = []
for image_id, count in sorted(bbox_counts.items()):
    if image_id > 0:
        count_prev = bbox_counts.get(image_id - 1, 0)
        # 
        if abs(count - count_prev) >= default: #1
            avg_values = average_bbox_counts.get(image_id)
            avg_values_prev = average_bbox_counts.get(image_id - 1)
            # 
            if avg_values and avg_values_prev and abs(avg_values[0] - avg_values_prev[1]) > default: #0.4
                selected_frames.append(image_id)

# extract images and annotations
selected_images = [img for img in coco_data["images"] if img["id"] in selected_frames]
selected_image_ids = {img["id"] for img in selected_images}
selected_annotations = [anno for anno in coco_data["annotations"] if anno["image_id"] in selected_image_ids]

# save new coco_files
new_coco_data = {
    "info": coco_data.get("info", {}),
    "licenses": coco_data.get("licenses", []),
    "categories": coco_data.get("categories", []),
    "images": selected_images,
    "annotations": selected_annotations
}
new_json_path = "/Path/to/new/json_file"
with open(new_json_path, "w") as file:
    json.dump(new_coco_data, file)

with open(new_json_path, "r") as file:
    new_coco_data = json.load(file)

selected_frame_ids = {img["id"] for img in new_coco_data["images"]}

video_path = "/Path/to/your/video_file/"
cap = cv2.VideoCapture(video_path)

# Output
output_dir = "/Path/to/selected_frames/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id in selected_frame_ids:
        frame_filename = f"frame_{frame_id}.jpg"
        output_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(output_path, frame)
    frame_id += 1

cap.release()