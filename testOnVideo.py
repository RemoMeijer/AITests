import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import cv2
from PIL import Image

# Define the number of classes: 1 (background) + number of object classes (crop, weed, etc.)
num_classes = 3  # Adjust this to the correct number of classes, e.g., 3 for background + crop + weed

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(pretrained=False)  # Set pretrained=False to avoid loading COCO weights

# Modify the classifier to match your number of classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights (state_dict) into the model
model_path = "plant_dect_fastercnn.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval()  # Set to evaluation mode

# Define the transformation (same as during training)
transform = lambda x: F.to_tensor(x).unsqueeze(0).to(device)

# Open the video file
video_path = "/home/remo/Afstudeerproject/AgronomischePerformanceMeting/test_video.mp4"
output_path = "output.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Define class labels (modify this based on your dataset)
class_labels = {1: "Crop", 2: "Weed"}  # Example classes

# Read and process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil)

    # Run inference
    with torch.no_grad():
        preds = model(img_tensor)

    # Extract predictions
    pred_boxes = preds[0]['boxes'].cpu().numpy()
    pred_scores = preds[0]['scores'].cpu().numpy()
    pred_labels = preds[0]['labels'].cpu().numpy()

    # Draw bounding boxes
    for i in range(len(pred_boxes)):
        if pred_scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, pred_boxes[i])
            label = class_labels.get(pred_labels[i], "Unknown")
            score = pred_scores[i]

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label text
            text = f"{label}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    # Display frame (optional)
    cv2.imshow("Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
