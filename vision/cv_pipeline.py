from ultralytics import YOLO
import cv2
import os
import csv

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input video
input_video = "data/videos/cctv2.webm"

# Output folder
os.makedirs("output", exist_ok=True)

# Open video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
output_video = "output/tracking_output2.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

# Evidence CSV
evidence_file = "output/tracking_evidence2.csv"

csv_file = open(evidence_file, "w", newline="")

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "timestamp",
    "camera_id",
    "track_id",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
])

print("Starting CASELINE tracking + evidence extraction...")

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # Draw tracking results
    annotated_frame = result.plot()

    # Extract tracking information
    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, cls, confidence in zip(
            boxes,
            track_ids,
            classes,
            confidences
        ):

            x1, y1, x2, y2 = box

            # Convert frame number to seconds
            timestamp = frame_number / fps

            # Get class name
            class_name = model.names[int(cls)]

            # Save evidence
            csv_writer.writerow([
                frame_number,
                round(timestamp, 2),
                "CAM01",
                int(track_id),
                class_name,
                round(float(confidence), 3),
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ])

    # Save video
    out.write(annotated_frame)

    # Display
    cv2.imshow("CASELINE - Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print("Tracking complete!")
print(f"Video saved to: {output_video}")
print(f"Evidence saved to: {evidence_file}")