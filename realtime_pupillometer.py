import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# -----------------------------
# 1. Model loading
# -----------------------------

def load_model(checkpoint_path, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# Same transform as training
transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------------
# 2. Quality checks
# -----------------------------

def check_eye_quality(eye_roi,
                      min_brightness=40,
                      max_brightness=215,
                      min_lap_var=80,
                      min_aspect_ratio=0.18):
    """
    Returns (is_ok, message) for a given eye ROI (BGR image).
    Tune thresholds based on your camera.
    """

    if eye_roi is None or eye_roi.size == 0:
        return False, "Invalid ROI"

    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()

    if mean_brightness < min_brightness:
        return False, f"Too dark ({mean_brightness:.1f}), increase light"
    if mean_brightness > max_brightness:
        return False, f"Too bright ({mean_brightness:.1f}), reduce light"

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < min_lap_var:
        return False, f"Too blurry (Var={lap_var:.1f}), hold still"

    h, w = gray.shape[:2]
    aspect_ratio = h / float(w + 1e-6)
    if aspect_ratio < min_aspect_ratio:
        return False, f"Eye may be too closed (AR={aspect_ratio:.2f})"

    return True, "Quality OK"

# -----------------------------
# 3. Prediction helper
# -----------------------------

def predict_pupil_diameter(model, eye_roi_bgr, device):
    # Convert BGR (OpenCV) → RGB (PIL)
    rgb = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor).item()

    return pred

# -----------------------------
# 4. Main realtime loop
# -----------------------------

def main():
    # ---- Config ----
    checkpoint_path = "./checkpoints/best_RealEsr_x2.pt"  # change if needed
    camera_index = 0
    ok_frames_required = 5

    # Distance thresholds (relative to frame width)
    min_eye_ratio = 0.10   # too far below this
    max_eye_ratio = 0.25   # too close above this

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if not Path(checkpoint_path).exists():
        print(f"Model checkpoint not found at {checkpoint_path}")
        return

    print("Loading model...")
    model = load_model(checkpoint_path, device)
    print("Model loaded.")

    # Load eye cascade
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    )
    if eye_cascade.empty():
        print("Could not load haarcascade_eye_tree_eyeglasses.xml")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open camera")
        return

    ok_counter = 0
    last_prediction = None

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        status_text = "No eye detected"

        if len(eyes) > 0:
            # choose the largest detected eye
            x, y, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
            eye_ratio = ew / w
            eye_roi = frame[y:y+eh, x:x+ew]

            # Draw eye rectangle
            cv2.rectangle(frame, (x, y), (x+ew, y+eh), (0, 255, 0), 2)

            # Distance checks
            if eye_ratio < min_eye_ratio:
                status_text = "Too far, move closer"
                ok_counter = 0
            elif eye_ratio > max_eye_ratio:
                status_text = "Too close, move back"
                ok_counter = 0
            else:
                # Quality checks
                is_ok, q_msg = check_eye_quality(eye_roi)
                status_text = q_msg

                if is_ok:
                    ok_counter += 1
                else:
                    ok_counter = 0

                # Stable, good-quality frames → run model
                if ok_counter >= ok_frames_required:
                    pred_mm = predict_pupil_diameter(model, eye_roi, device)
                    last_prediction = pred_mm
                    print(f"Predicted pupil diameter: {pred_mm:.3f} mm")
                    ok_counter = 0

        # Overlay text
        cv2.putText(frame, status_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if last_prediction is not None:
            cv2.putText(frame, f"Pupil: {last_prediction:.3f} mm",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Realtime Pupillometer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()