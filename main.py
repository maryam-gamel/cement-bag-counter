import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Video
cap = cv2.VideoCapture("vid2.mp4")

# Resize dimensions
WIDTH = 1020
HEIGHT = 500

# Output video
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "output_counted.mp4",
    fourcc,
    fps,
    (WIDTH, HEIGHT)
)

# Counting line
LINE_Y = 300

# Keep track of previous Y position for each bag
previous_positions = {}

# IDs that have already been counted
counted_ids = set()

# Total bags
total_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Tracking
    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.08,
        iou=0.5,
        classes=[0],
        imgsz=1280,
        verbose=False
    )

    # Draw counting line
    cv2.line(
        frame,
        (300, LINE_Y),
        (700, LINE_Y),
        (255, 0, 255),
        3
    )
    # Check detections
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = map(int, box)

            # Center of bag
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Draw ID
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # Check previous position
            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                # Bag moving from bottom to top
                if (
                    previous_y > LINE_Y
                    and center_y <= LINE_Y
                    and track_id not in counted_ids
                ):

                    total_count += 1
                    counted_ids.add(track_id)

                    print(f"Bag counted! Total = {total_count}")

            # Save current position
            previous_positions[track_id] = center_y

    # Display total
    cv2.putText(
        frame,
        f"Total Bags: {total_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Save processed frame
    out.write(frame)

    # Show frame
    cv2.imshow("RGB", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
