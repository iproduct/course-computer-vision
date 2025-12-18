import cv2
import numpy as np
import tensorflow as tf

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load model (weights must be trained already)
model = tf.keras.models.load_model("saved/cifar10_cnn.model.keras")
# OR if loading weights:
# model.load_weights("cifar10_cnn_weights.h5")

# Start camera
cap = cv2.VideoCapture(0)

print("Starting camera... press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame
    img = cv2.resize(frame, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Predict
    predictions = model.predict(img_input, verbose=0)
    label_index = np.argmax(predictions)
    label = class_names[label_index]

    # Display prediction
    cv2.putText(frame, f"Prediction: {label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("CIFAR-10 Live Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
