import cv2
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow import keras
import matplotlib.pyplot as plt
import tempfile

class DeepfakeDetector:
    def __init__(self, model_path, img_size=224, max_frames=30):
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size
        self.max_frames = max_frames
        self.explainer = lime_image.LimeImageExplainer()

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = frame.astype("float32") / 255.0
        return frame

    def explain_frame(self, frame):
        def predict_fn(images):
            images = np.array(images) / 255.0
            return self.model.predict(images)

        explanation = self.explainer.explain_instance(
            frame, predict_fn, top_labels=1, hide_color=0, num_samples=1000
        )
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            top_label, positive_only=False, num_features=10, hide_rest=False
        )
        out_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        plt.imsave(out_path, mark_boundaries(temp / 255.0, mask))
        return out_path

    def predict_video(self, video_path, explain=False):
        cap = cv2.VideoCapture(video_path)
        preds = []
        lime_path = None
        count = 0

        while cap.isOpened() and count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = self.preprocess_frame(frame)
            frame_input = np.expand_dims(frame_resized, axis=0)
            pred = self.model.predict(frame_input, verbose=0)[0][0]
            preds.append(pred)

            if explain and count == 10:
                lime_path = self.explain_frame((frame_resized * 255).astype(np.uint8))

            count += 1

        cap.release()
        avg_score = np.mean(preds) if preds else 0.0
        label = "FAKE" if avg_score > 0.5 else "REAL"
        return {"label": label, "confidence": float(avg_score), "lime_path": lime_path}
# Model\resnet50_model_ep10.keras

detector = DeepfakeDetector("Model/resnet50_model_ep10.keras")
result = detector.predict_video("Demo_video/Real/aktnlyqpah.mp4", explain=True)
print(result)
# -> {'label': 'FAKE', 'confidence': 0.78, 'lime_path': '/tmp/tmpabcd.png'}
