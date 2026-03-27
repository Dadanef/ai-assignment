from fer import FER
import cv2

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

detector = FER(mtcnn=True)


def classify_emotions(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_emotions(image_rgb)

    faces_data = []

    for idx, face in enumerate(results, start=1):
        box = face["box"]
        emotions = face["emotions"]

        full_scores = {e: float(emotions.get(e, 0.0)) for e in EMOTIONS}

        top_emotion = max(full_scores, key=full_scores.get)
        top_score = full_scores[top_emotion]

        faces_data.append({
            "face_id": idx,
            "box": {
                "x": int(box[0]),
                "y": int(box[1]),
                "w": int(box[2]),
                "h": int(box[3]),
            },
            "emotions": full_scores,
            "top_emotion": top_emotion,
            "top_score": top_score
        })

    return faces_data