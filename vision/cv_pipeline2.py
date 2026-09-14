from ultralytics import YOLO
import cv2
import os
import csv

# ==========================================
# 1. LOAD MODEL
# ==========================================

model = YOLO("yolo11s.pt")


# ==========================================
# 2. INPUT / OUTPUT
# ==========================================

input_video = "data/videos/cctv2.webm"

os.makedirs("output", exist_ok=True)

output_video = "output/tracking_v2.mp4"
output_csv = "output/tracking_evidence_v2.csv"


# ==========================================
# 3. OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ==========================================
# 4. VIDEO WRITER
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)


# ==========================================
# 5. CSV EVIDENCE FILE
# ==========================================

csv_file = open(
    output_csv,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "timestamp",
    "camera_id",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
])


# ==========================================
# 6. LOW-LIGHT ENHANCEMENT
# ==========================================

def enhance_low_light(frame):

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced_lab = cv2.merge((l, a, b))

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# ==========================================
# 7. PROCESS VIDEO
# ==========================================

print("Starting CASELINE V2...")
print("Low-light enhancement: ON")
print("Person-only detection: ON")
print("ByteTrack: ON")
print("Evidence CSV: ON")


frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # --------------------------------------
    # STEP A: Enhance frame
    # --------------------------------------

    enhanced_frame = enhance_low_light(frame)


    # --------------------------------------
    # STEP B: YOLO + ByteTrack
    # --------------------------------------

    results = model.track(
        enhanced_frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=0.55,

        iou=0.45,

        classes=[0],

        verbose=False
    )


    result = results[0]


    # --------------------------------------
    # STEP C: SAVE PERSON EVIDENCE
    # --------------------------------------

    if result.boxes is not None and result.boxes.id is not None:

        track_ids = result.boxes.id.cpu().numpy().astype(int)

        boxes = result.boxes.xyxy.cpu().numpy()

        confidences = result.boxes.conf.cpu().numpy()


        for track_id, box, confidence in zip(
            track_ids,
            boxes,
            confidences
        ):

            x1, y1, x2, y2 = box

            timestamp = frame_number / fps

            csv_writer.writerow([
                frame_number,
                round(timestamp, 2),
                "CAM01",
                track_id,
                round(float(confidence), 3),
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ])


    # --------------------------------------
    # STEP D: DRAW RESULTS
    # --------------------------------------

    annotated_frame = result.plot()


    # --------------------------------------
    # STEP E: DISPLAY
    # --------------------------------------

    cv2.imshow(
        "CASELINE - V2 Tracking",
        annotated_frame
    )


    # --------------------------------------
    # STEP F: SAVE VIDEO
    # --------------------------------------

    out.write(annotated_frame)


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# 8. CLEANUP
# ==========================================

cap.release()
out.release()
csv_file.close()

cv2.destroyAllWindows()


print("\n================================")
print("CASELINE V2 COMPLETE!")
print("================================")

print(f"Video saved to: {output_video}")
print(f"Evidence saved to: {output_csv}")