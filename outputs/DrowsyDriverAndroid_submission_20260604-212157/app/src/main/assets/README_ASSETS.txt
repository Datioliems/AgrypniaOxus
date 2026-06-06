Place these files here before running the full AI pipeline:

1. face_landmarker.task
   Included. Source:
   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

2. drowsiness_model.tflite
   Optional CNN model. The app can run a landmark-based prototype without it,
   then use this model once your training pipeline exports TensorFlow Lite.
