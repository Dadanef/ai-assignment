import argparse
import os

from partner_a import run_preprocessing_demo, run_face_detection_tuning, run_webcam_mode
from classifier import classify_emotions
from visualize import save_annotated_image, save_summary_figure
from batch_analysis import run_batch_analysis


def ensure_output_dirs():
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/a1", exist_ok=True)
    os.makedirs("output/a2", exist_ok=True)
    os.makedirs("output/b2", exist_ok=True)
    os.makedirs("output/b3", exist_ok=True)


def run_classify_mode(image_path: str):
    faces = classify_emotions(image_path)

    print(f"Detected faces: {len(faces)}")
    for face in faces:
        print(f"\nFace {face['face_id']}")
        print("Box:", face["box"])
        print("Top emotion:", face["top_emotion"])
        print("Top score:", round(face["top_score"] * 100, 2), "%")
        print("All scores:")
        for emotion, score in face["emotions"].items():
            print(f"  {emotion}: {score:.4f}")


def run_visualize_mode(image_path: str):
    faces = classify_emotions(image_path)

    annotated_path = "output/b2/annotated_faces.png"
    summary_path = "output/b2/summary_figure.png"

    save_annotated_image(image_path, faces, annotated_path)
    save_summary_figure(image_path, faces, summary_path)

    print(f"Saved: {annotated_path}")
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Facial Emotion Recognition Project"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["a1", "a2", "webcam", "classify", "visualize", "batch"],
        help="Run mode"
    )
    parser.add_argument(
        "--image",
        default="images/test.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--dataset",
        default="labeled_dataset",
        help="Path to labeled dataset"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=3,
        help="Process every N-th frame in webcam mode"
    )

    args = parser.parse_args()
    ensure_output_dirs()

    if args.mode == "a1":
        run_preprocessing_demo(args.image, output_dir="output/a1")
    elif args.mode == "a2":
        run_face_detection_tuning(args.image, output_dir="output/a2")
    elif args.mode == "webcam":
        run_webcam_mode(skip_every_n=args.skip, save_dir="output/webcam_shots")
    elif args.mode == "classify":
        run_classify_mode(args.image)
    elif args.mode == "visualize":
        run_visualize_mode(args.image)
    elif args.mode == "batch":
        run_batch_analysis(dataset_dir=args.dataset, output_dir="output/b3")


if __name__ == "__main__":
    main()