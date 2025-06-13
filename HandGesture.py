import os
import zipfile
import urllib.request
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def download_dataset():
    url = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/master.zip"
    dataset_dir = "dataset"
    zip_path = "dataset.zip"

    if not os.path.exists(dataset_dir):
        print("📥 Mengunduh dataset...")
        urllib.request.urlretrieve(url, zip_path)

        print("📦 Mengekstrak dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

        os.remove(zip_path)
        print("✅ Dataset siap!")
    else:
        print("✅ Dataset sudah tersedia.")


def load_data():
    images = []
    labels = []

    data_path = "dataset/Sign-Language-Digits-Dataset-master/Dataset"
    for label in range(1, 6):  # gesture 1-5 saja
        folder = os.path.join(data_path, str(label))
        if not os.path.exists(folder):
            print(f"⚠️ Folder {folder} tidak ditemukan!")
            continue
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Gagal baca gambar: {img_path}")
                continue
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(label - 1)

    images = np.array(images).reshape(-1, 64, 64, 1).astype("float32") / 255.0
    labels = to_categorical(labels, num_classes=5)
    return train_test_split(images, labels, test_size=0.2, random_state=42)


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def predict_frame(frame, model):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, 64, 64, 1).astype("float32") / 255.0
    pred = model.predict(img, verbose=0)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]
    return class_idx + 1, confidence


def start_webcam_with_mediapipe(model):
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7) as hands:

        print("📷 Tekan 'q' untuk keluar dari webcam.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min = int(min(x_coords) * w) - 20
                x_max = int(max(x_coords) * w) + 20
                y_min = int(min(y_coords) * h) - 20
                y_max = int(max(y_coords) * h) + 20

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, w)
                y_max = min(y_max, h)

                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size > 0:
                    gesture, conf = predict_frame(hand_img, model)
                    label = f"Gesture: {gesture} ({conf*100:.1f}%)"
                    cv2.putText(frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                cv2.putText(frame, "Tangan tidak terdeteksi", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Hand Gesture Recognition with MediaPipe", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    download_dataset()
    x_train, x_test, y_train, y_test = load_data()

    if not os.path.exists("gesture_model.h5"):
        print("⚙️ Melatih model...")
        model = build_model()
        model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
        model.save("gesture_model.h5")
        print("✅ Model disimpan ke gesture_model.h5")
    else:
        print("📂 Memuat model dari file...")
        model = load_model("gesture_model.h5")

    start_webcam_with_mediapipe(model)


if __name__ == "__main__":
    main()
