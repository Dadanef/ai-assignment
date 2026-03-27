import cv2
import os
import time
import pandas as pd

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def preprocess_image(img, target_size=(600, 600), clip_limit=2.0, tile_grid_size=(8, 8)):
    if img is None:
        raise ValueError("Input image is None")

    resized = cv2.resize(img, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    clahe_gray = clahe.apply(gray)

    return {
        "resized_bgr": resized,
        "gray": gray,
        "clahe_gray": clahe_gray
    }


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    return img


def detect_faces_haar(gray_image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier")

    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )
    filtered_faces = []

    for (x, y, w, h) in faces:
        if w > 80 and h > 80:  # фильтр
            filtered_faces.append((x, y, w, h))

    faces = filtered_faces

    def remove_duplicates(faces):
        result = []
        for (x, y, w, h) in faces:
            keep = True
            for (x2, y2, w2, h2) in result:
                if abs(x - x2) < 30 and abs(y - y2) < 30:
                    keep = False
            if keep:
                result.append((x, y, w, h))
        return result

    faces = remove_duplicates(faces)
    return faces



def draw_faces(image, faces, label_prefix="Face"):
    output = image.copy()

    for idx, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{label_prefix} {idx}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output


def run_preprocessing_demo(image_path, output_dir="output/a1"):
    os.makedirs(output_dir, exist_ok=True)

    img = load_image(image_path)
    processed = preprocess_image(img)

    cv2.imwrite(os.path.join(output_dir, "01_resized.jpg"), processed["resized_bgr"])
    cv2.imwrite(os.path.join(output_dir, "02_gray.jpg"), processed["gray"])
    cv2.imwrite(os.path.join(output_dir, "03_clahe_gray.jpg"), processed["clahe_gray"])

    print("A1 preprocessing completed.")
    print(f"Saved outputs to: {output_dir}")


def run_face_detection_tuning(image_path, output_dir="output/a2"):
    os.makedirs(output_dir, exist_ok=True)

    img = load_image(image_path)
    processed = preprocess_image(img)
    gray = processed["clahe_gray"]

    parameter_combos = [
        {"name": "combo_1", "scale_factor": 1.03, "min_neighbors": 2, "min_size": (20, 20)},
        {"name": "combo_2", "scale_factor": 1.05, "min_neighbors": 3, "min_size": (30, 30)},
        {"name": "combo_3", "scale_factor": 1.10, "min_neighbors": 5, "min_size": (40, 40)},
        {"name": "combo_4", "scale_factor": 1.20, "min_neighbors": 6, "min_size": (50, 50)},
    ]

    rows = []

    for combo in parameter_combos:
        try:
            faces = detect_faces_haar(
                gray,
                scale_factor=combo["scale_factor"],
                min_neighbors=combo["min_neighbors"],
                min_size=combo["min_size"]
            )

            annotated = draw_faces(processed["resized_bgr"], faces, label_prefix=combo["name"])
            save_path = os.path.join(output_dir, f"{combo['name']}.jpg")
            cv2.imwrite(save_path, annotated)

            rows.append({
                "combo_name": combo["name"],
                "scale_factor": combo["scale_factor"],
                "min_neighbors": combo["min_neighbors"],
                "min_size": str(combo["min_size"]),
                "faces_detected": len(faces),
                "status": "success",
                "notes": "OK"
            })

        except Exception as e:
            rows.append({
                "combo_name": combo["name"],
                "scale_factor": combo["scale_factor"],
                "min_neighbors": combo["min_neighbors"],
                "min_size": str(combo["min_size"]),
                "faces_detected": 0,
                "status": "error",
                "notes": str(e)
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "tuning_results.csv")
    df.to_csv(csv_path, index=False)

    print("A2 face detection tuning completed.")
    print(df)
    print(f"Saved table to: {csv_path}")


def run_webcam_mode(skip_every_n=3, save_dir="output/webcam_shots"):
    os.makedirs(save_dir, exist_ok=True)

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not open webcam")

    prev_time = time.time()
    frame_count = 0
    last_faces = []

    print("Press 's' to save screenshot, 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Warning: failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % skip_every_n == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            last_faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )

        display_frame = draw_faces(frame, last_faces)

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press s = save | q = quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("A3 - Real-time Webcam", display_frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = os.path.join(save_dir, f"screenshot_{int(time.time())}.jpg")
            cv2.imwrite(filename, display_frame)
            print(f"Screenshot saved: {filename}")

    cam.release()
    cv2.destroyAllWindows()