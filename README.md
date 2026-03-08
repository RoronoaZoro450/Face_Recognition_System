# Face Recognition App

A Python-based desktop application that performs **real-time face recognition using a webcam**.
The application detects faces from a live camera feed, converts them into embeddings, and compares them with stored embeddings in a database to identify individuals.

The system uses **computer vision and deep learning-based face embeddings** to recognize faces accurately.

---

# Features

* Real-time webcam face detection
* Face recognition using embedding comparison
* SQLite database for storing known face embeddings
* PySide6 GUI interface
* Modular architecture for easy scalability
* Automatic face matching using vector similarity

---

# Tech Stack

* Python
* OpenCV
* face_recognition / deep face embedding model
* NumPy
* SQLite
* PySide6

---

# Project Structure

```
FaceRecognitionApp/
│
├── assets/
│   └── icon/
│       └── app_icon.png        # Application icon
│
├── core/
│   ├── Embedding.py            # Generates face embeddings
│   ├── Inference.py            # Performs face recognition
│   └── SQliteDB.py             # Handles database operations
│
├── ui/
│   └── App.ui                  # GUI layout
│
├── App.py                      # Main application entry point
├── requirements.txt            # Project dependencies
└── .gitignore
```

---

# How the Application Works (Behind the Scenes)

The application follows a **three-stage face recognition pipeline**.

---

## 1. Face Embedding Generation

This stage is handled by:

```
core/Embedding.py
```

Process:

1. The system captures or loads a face image.
2. The face is detected using a face detection model.
3. The detected face is converted into a **face embedding**.

A **face embedding** is a numerical vector (usually 128 or 512 dimensions) that represents unique facial features.

Example representation:

```
[0.21, -0.14, 0.88, 0.03, ...]
```

These embeddings act like **digital fingerprints for faces**.

---

## 2. Storing Face Embeddings

Handled by:

```
core/SQliteDB.py
```

The application stores:

* Person name
* Face embedding vector

Inside a **SQLite database**.

Example database structure:

```
ID | Name | Face_Embedding
```

This allows the system to **quickly retrieve and compare faces** during recognition.

---

## 3. Real-Time Face Recognition

Handled by:

```
core/Inference.py
```

When the application runs:

1. The webcam captures frames continuously.
2. Each frame is scanned for faces.
3. For every detected face:

   * A new embedding is generated.
4. The embedding is compared with stored embeddings in the database.

The comparison uses **vector distance (usually Euclidean distance)**.

```
Small distance → Same person
Large distance → Different person
```

If the distance is below a threshold, the person is recognized.

---

## 4. GUI Interaction

Handled by:

```
ui/App.ui
App.py
```

The GUI:

* Starts the camera
* Displays live video feed
* Shows bounding boxes around detected faces
* Displays the recognized person's name

All recognition happens **in real-time while the camera is running**.

---

# Installation

Install required packages:

```
pip install -r requirements.txt
```

Run the application:

```
python App.py
```

---

# Future Improvements

* Face mask detection
* Multi-person tracking
* Cloud face database
* Attendance system integration
* GPU acceleration for faster recognition

---

# Author

Nirbhay Shegale
