import cv2
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
VIDEO_PATH = r"data\People, Street, Ukraine. Free Stock Video.mp4"  # Or use 0 for webcam
MODEL_PATH = "yolo11n.pt"  # Try 'yolo11s.pt' or 'yolo11m.pt' for better accuracy
THRESHOLD = 50  # Alert threshold
CONFIDENCE = 0.10  # Very low - detects much more (may include false positives)
IOU_THRESHOLD = 0.5  # Higher to avoid double-counting overlapping people

# COCO class names for reference
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Colors (BGR format)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)


class CrowdCounter:
    def __init__(self, model_path, threshold=50, confidence=0.10, iou_threshold=0.5):
        self.threshold = threshold
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_path)

    def process_frame(self, frame, debug=False):
        """
        Process a single frame for crowd detection and counting.
        """
        # Run YOLOv11 detection WITHOUT class filter first
        results = self.model(
            frame,
            conf=self.confidence,  # Confidence threshold
            iou=self.iou_threshold,  # NMS IOU threshold
            verbose=False
        )

        person_count = 0
        all_detections = []

        if results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu()  # Get boxes in xywh format
            confidences = results[0].boxes.conf.cpu()  # Get confidence scores
            class_ids = results[0].boxes.cls.cpu().int().tolist()  # Get class IDs

            # Count people and gather all detections
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detection_info = {
                    'box': box,
                    'conf': conf,
                    'cls_id': cls_id,
                    'cls_name': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                }
                all_detections.append(detection_info)

                if cls_id == 0:  # Class 0 is 'person'
                    person_count += 1

            # Draw ALL detections in debug mode
            if debug:
                for det in all_detections:
                    box = det['box']
                    conf = det['conf']
                    cls_name = det['cls_name']
                    is_person = det['cls_id'] == 0

                    x, y, w, h = box
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)

                    # Use green for people, yellow for other objects
                    color = GREEN if is_person else YELLOW
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label with class name and confidence
                    label = f"{cls_name}: {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
            else:
                # Only draw people in normal mode
                for det in all_detections:
                    if det['cls_id'] == 0:  # Only people
                        box = det['box']
                        x, y, w, h = box
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)

        # Draw UI overlay
        self.draw_ui(frame, person_count, all_detections if debug else None)

        return frame, person_count, all_detections

    def draw_ui(self, frame, count, all_detections=None):
        """
        Draw UI overlay with counter and threshold alert.
        """
        # Background panel for counter
        panel_height = 100 if all_detections is None else 150
        cv2.rectangle(frame, (10, 10), (350, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, panel_height), WHITE, 2)

        # Counter
        cv2.putText(
            frame,
            f"People Count: {count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            WHITE,
            2
        )

        # Threshold alert
        alert_color = RED if count >= self.threshold else WHITE
        alert_text = f"Threshold: {count}/{self.threshold}"

        cv2.putText(
            frame,
            alert_text,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            alert_color,
            2
        )

        # Show total detections in debug mode
        if all_detections is not None:
            cv2.putText(
                frame,
                f"Total Objects: {len(all_detections)}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                YELLOW,
                2
            )

            # Count by class
            class_counts = {}
            for det in all_detections:
                cls_name = det['cls_name']
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            y_offset = 145
            for cls_name, cls_count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                cv2.putText(
                    frame,
                    f"{cls_name}: {cls_count}",
                    (400, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    YELLOW,
                    2
                )
                y_offset += 25

        # Draw warning box if threshold exceeded
        if count >= self.threshold:
            cv2.rectangle(
                frame,
                (frame.shape[1]//2 - 200, 50),
                (frame.shape[1]//2 + 200, 100),
                (0, 0, 255),
                -1
            )
            cv2.rectangle(
                frame,
                (frame.shape[1]//2 - 200, 50),
                (frame.shape[1]//2 + 200, 100),
                WHITE,
                2
            )
            cv2.putText(
                frame,
                "THRESHOLD EXCEEDED!",
                (frame.shape[1]//2 - 160, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                WHITE,
                2
            )


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}")
        return

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    video_writer = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # Initialize crowd counter
    counter = CrowdCounter(
        model_path=MODEL_PATH,
        threshold=THRESHOLD,
        confidence=CONFIDENCE,
        iou_threshold=IOU_THRESHOLD
    )

    print(f"Starting crowd counting with threshold: {THRESHOLD}")
    print(f"Model: {MODEL_PATH} | Confidence: {CONFIDENCE} | IOU: {IOU_THRESHOLD}")
    print("="*60)
    print("DEBUG MODE ON - First 30 frames will show ALL detections")
    print("Green boxes = People | Yellow boxes = Other objects")
    print("="*60)
    print("Press 'q' to quit")

    # Main processing loop
    frame_count = 0
    max_count = 0
    debug_frames = 30  # Show debug info for first 30 frames

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process frame (with debug for first few frames)
        debug = frame_count < debug_frames
        frame, count, all_detections = counter.process_frame(frame, debug=debug)

        # Track maximum count
        if count > max_count:
            max_count = count

        # Write and display
        video_writer.write(frame)
        cv2.imshow("Crowd Management - YOLOv11", frame)

        # Print detailed info for debug frames
        frame_count += 1
        if frame_count <= debug_frames:
            print(f"\n=== FRAME {frame_count} ===")
            print(f"People detected: {count}")
            if all_detections:
                print(f"Total objects: {len(all_detections)}")
                # Show top 5 detections
                class_counts = {}
                for det in all_detections:
                    cls_name = det['cls_name']
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                print("Objects detected:")
                for cls_name, cls_count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {cls_name}: {cls_count}")
        elif frame_count % 30 == 0:
            print(f"Frame {frame_count}: {count} people | Max so far: {max_count}")

        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print("=== Final Results ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Maximum people detected: {max_count}")
    print(f"Threshold was {'EXCEEDED' if max_count >= THRESHOLD else 'NOT exceeded'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
