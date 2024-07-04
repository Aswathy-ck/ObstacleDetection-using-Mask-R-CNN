import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from torchvision import models, transforms

root = tk.Tk()
root.title("Obstacles Detection")

label = tk.Label(root)
label.pack()

# Load the pre-trained Mask R-CNN model
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the transformation to apply to the frames for the model
transform = transforms.Compose([
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

drone_height = 20
drone_width = 20

def update_frame():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    # Convert frame to RGB and apply transformations
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

    # Get model outputs
    boxes = outputs[0]['boxes'].cpu().numpy()
    masks = outputs[0]['masks'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    top_left_obstacle = False
    top_right_obstacle = False
    bottom_left_obstacle = False
    bottom_right_obstacle = False
    stop_due_to_size = False

    frame_height, frame_width, _ = frame.shape
    section_width = frame_width // 2
    section_height = frame_height // 2

    # Draw the lines dividing the frame into four segments
    cv2.line(frame, (section_width, 0), (section_width, frame_height), (255, 0, 0), 2)
    cv2.line(frame, (0, section_height), (frame_width, section_height), (255, 0, 0), 2)

    # Initialize cumulative obstruction values
    top_left_obstruction = 0
    top_right_obstruction = 0
    bottom_left_obstruction = 0
    bottom_right_obstruction = 0

    for i in range(len(boxes)):
        if scores[i] < 0.5:
            continue

        box = boxes[i]
        mask = masks[i][0]
        mask = (mask > 0.5).astype("uint8") * 255
        mask_area = np.sum(mask) // 255  # Total number of non-zero pixels in the mask
        frame_area = frame_height * frame_width

        # Check if the mask area is 80% or more of the frame area
        if mask_area >= 0.8 * frame_area:
            stop_due_to_size = True

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        # Determine the percentage of obstruction in each quadrant
        top_left_mask = mask[:section_height, :section_width]
        top_right_mask = mask[:section_height, section_width:]
        bottom_left_mask = mask[section_height:, :section_width]
        bottom_right_mask = mask[section_height:, section_width:]

        top_left_obstruction += np.sum(top_left_mask) // 255
        top_right_obstruction += np.sum(top_right_mask) // 255
        bottom_left_obstruction += np.sum(bottom_left_mask) // 255
        bottom_right_obstruction += np.sum(bottom_right_mask) // 255

        # Extract and display the coordinates, height, and width of the bounding boxes
        x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
        obstacle_width = x2 - x1
        obstacle_height = y2 - y1

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Display the coordinates, height, and width
        label_text = f"({x1},{y1}) W:{obstacle_width}px H:{obstacle_height}px"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate percentages for each quadrant
    total_top_left = section_width * section_height
    total_top_right = section_width * section_height
    total_bottom_left = section_width * section_height
    total_bottom_right = section_width * section_height

    top_left_percentage = (top_left_obstruction / total_top_left) * 100
    top_right_percentage = (top_right_obstruction / total_top_right) * 100
    bottom_left_percentage = (bottom_left_obstruction / total_bottom_left) * 100
    bottom_right_percentage = (bottom_right_obstruction / total_bottom_right) * 100

    # Ensure percentages do not exceed 100%
    top_left_percentage = min(top_left_percentage, 100)
    top_right_percentage = min(top_right_percentage, 100)
    bottom_left_percentage = min(bottom_left_percentage, 100)
    bottom_right_percentage = min(bottom_right_percentage, 100)

    # Update obstacle status based on percentages
    top_left_obstacle = top_left_percentage > 0
    top_right_obstacle = top_right_percentage > 0
    bottom_left_obstacle = bottom_left_percentage > 0
    bottom_right_obstacle = bottom_right_percentage > 0

    # Display which quadrants have obstacles and their percentages
    if top_left_obstacle:
        cv2.putText(frame, f"Top Left Obstructed: {top_left_percentage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Top Left Clear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if top_right_obstacle:
        cv2.putText(frame, f"Top Right Obstructed: {top_right_percentage:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Top Right Clear", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if bottom_left_obstacle:
        cv2.putText(frame, f"Bottom Left Obstructed: {bottom_left_percentage:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Bottom Left Clear", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if bottom_right_obstacle:
        cv2.putText(frame, f"Bottom Right Obstructed: {bottom_right_percentage:.2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Bottom Right Clear", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if stop_due_to_size:
        cv2.putText(frame, "Stop: Large Obstacle", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
