To run this program successfully, you need to make sure the following Python libraries are installed. You can install them using pip in your terminal or command prompt.

✅ Required Libraries:
-NumPy
-OpenCV (cv2)
-TensorFlow
-scikit-learn
-MediaPipe

🔧 Installation Commands:

```python
pip install numpy
pip install opencv-python
pip install tensorflow
pip install scikit-learn
pip install mediapipe
```
📝 Notes:
Make sure your Python version is 3.7 to 3.10, since mediapipe may not fully support newer versions yet.

You also need a working webcam for the real-time detection to work.
how this code run:
🧠 Purpose
The program detects and classifies hand gestures (numbers 1 to 5) in real-time using a webcam. It uses deep learning (CNN) and MediaPipe for hand tracking.

⚙️ How It Works
Dataset Downloading
The function download_dataset() downloads a dataset of hand gesture images (1–5) from GitHub and extracts it for training.

Data Loading & Preprocessing
load_data() reads grayscale images from the dataset, resizes them to 64x64 pixels, normalizes pixel values, and converts labels to one-hot format.

Model Building
build_model() creates a Convolutional Neural Network (CNN) with:

2 convolution layers

2 max-pooling layers

1 flatten layer

2 dense layers (ending with softmax)

Model Training
In the main() function:

If a model file gesture_model.h5 doesn’t exist, it trains the model on the dataset.

If it exists, it simply loads the trained model.

Real-Time Prediction (Webcam + MediaPipe)
start_webcam_with_mediapipe(model):

Opens the webcam and uses MediaPipe to detect the hand.

Extracts the region around the hand and sends it to the trained model.

Predicts which gesture (1–5) is being shown and displays it on the screen with confidence score.

🔁 Summary Flow:
Download → Preprocess → Train/Load Model → Webcam Input → Hand Detection → Predict Gesture → Display
