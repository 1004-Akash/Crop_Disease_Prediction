# 🌿 Crop Disease Detection using CNN

A deep learning project to classify crop leaf diseases into categories like **Healthy**, **Rust**, and **Powdery Mildew**, using image data and a Convolutional Neural Network (CNN). This model supports early detection of plant diseases to assist farmers in making timely decisions.

---

## 📁 Dataset Structure

The dataset is organized as follows:

```
Dataset/
├── Train/
│   ├── Healthy/
│   ├── Rust/
│   └── Powdery/
├── Validation/
└── Test/
```

---

## 🧪 Sample Image Viewer

Images were read and verified using Python's `os` and `PIL` modules:

```python
from PIL import Image
from IPython.display import display

image_path = 'Dataset/Train/Healthy/sample.jpg'
with open(image_path, 'rb') as f:
    display(Image.open(f).resize((400, 400)))
```

---

## 🧐 Model Architecture

Framework: **TensorFlow / Keras**
Image size: **225 x 225**

Preprocessing: `ImageDataGenerator` with:

* Rescaling
* Shearing
* Zooming
* Horizontal flipping

Classification mode: **Categorical (3 classes)**

```python
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
```

---

## 🏋️ Training

```python
train_generator = train_datagen.flow_from_directory(
    'Dataset/Train/Train',
    target_size=(225, 225),
    batch_size=32,
    class_mode='categorical'
)
```

* Training on \~1000+ images
* Validation set used to monitor overfitting

### ✅ Accuracy:

* **Training Accuracy**: \~91%
* **Validation Accuracy**: \~80%

---

## 📊 Results & Evaluation

* Model is performing well for classification of 3 classes
* To be added:

  * Confusion matrix
  * Accuracy/Loss plots

---

## 🚀 Future Scope

* Convert model to `.tflite` for edge/mobile deployment
* Deploy as a **web** or **mobile** application
* Add support for real-time camera or drone feeds
* Integrate with IoT agriculture systems

---

## 🧰 Tech Stack

* Python 3.10
* TensorFlow / Keras
* Pillow, Matplotlib
* Jupyter Notebook

---

## 🤝🏼 Author

**Akash Elayaraja**
B.Tech Artificial Intelligence & Data Science
RMK Engineering College

[GitHub](https://github.com/1004-Akash) • [LinkedIn](https://www.linkedin.com/in/akash-elayaraja-18842327a/)

---

⭐ *If you found this useful, give the repo a star!*
