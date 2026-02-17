# ========Imports========
import sys
import cv2
import numpy as np
import os
import shutil
from datetime import datetime

from Embedding import Embedding # Embedding generator module
from Inference import FaceMatcher # Face matcher module

from PySide6.QtWidgets import QApplication, QLabel, QPushButton,QLineEdit, QStackedWidget, QMessageBox
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap


# ========= APP WINDOW ==========

class AppWindow:
    def __init__(self):
        loader = QUiLoader()
        ui_file = QFile("App.ui")
        ui_file.open(QFile.ReadOnly)
        self.window = loader.load(ui_file)
        ui_file.close()

        # =============UI============ 
        # Getting all UI elements
        self.stack = self.window.findChild(QStackedWidget, "stack")
        self.camera_label = self.window.findChild(QLabel, "feed")
        self.camera_label_reg = self.window.findChild(QLabel, "Reg_feed")
        self.detected_name = self.window.findChild(QLabel, "DetectedName")
        self.capture_counter_reg = self.window.findChild(QLabel, "Captured_Images_Reg")

        self.register_login_btn = self.window.findChild(QPushButton, "RegisterLogin")
        self.capture_login_btn = self.window.findChild(QPushButton, "CaptureLogin")
        self.capture_reg_btn = self.window.findChild(QPushButton, "Capture_Register")
        self.register_person = self.window.findChild(QPushButton, "Register_Reg")

        self.name_input = self.window.findChild(QLineEdit, "Name_Input")
        self.id_input = self.window.findChild(QLineEdit, "Id_Input")
        self.next_btn = self.window.findChild(QPushButton, "Next")
        self.back_to_login = self.window.findChild(QPushButton, "Back_Next_2")
        self.back_to_next = self.window.findChild(QPushButton, "Back_Reg")

        self.stack.setCurrentIndex(0)

        # Runtime Initialization 
        self.last_frame = None
        self.current_bbox = None
        self.current_name = None
        self.current_sim = None
        self.capture_score = 1
        self.inference_started = False

        # Camera Thread 
        self.worker1 = CameraThread()                                # initializing camera thread
        self.worker1.Img_npArray.connect(self.update_frame_buffer)   # Getting signal to update frame buffer for inference
        self.worker1.start()                                         # Starting camera thread  

        # Face Matcher 
        self.matcher = FaceMatcher(
            emb_root = os.path.join("Data", "embeddings")
        )

        # Inference Thread 
        self.inf_worker = InferenceWorker(self.matcher)              # initializing inference thread
        self.inf_worker.bboxs.connect(self.update_bbox)              # Getting signal to update bounding box coordinates
        self.inf_worker.result.connect(self.update_result)           # Getting signal to update similarity score and detected name

        # Buttons 
        self.register_login_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))  # Moving to registration page
        self.next_btn.clicked.connect(self.UserDetail)                                  # Calling UserDetail function to validate user details and move to capture page
        self.back_to_login.clicked.connect(lambda: self.stack.setCurrentIndex(0))       # Moving back to login page
        self.back_to_next.clicked.connect(lambda: self.stack.setCurrentIndex(1))        # Moving back to details page

        self.capture_login_btn.clicked.connect(self.capture_Login)                      # Calling capture_Login function to capture login details and show message box
        self.register_person.clicked.connect(self.EmbeddingGenerator)                   # Calling EmbeddingGenerator function to generate embedding from captured images and save to disk
        self.capture_reg_btn.clicked.connect(self.Reg_Capture)                          # Calling Reg_Capture function to capture registration images and save to disk

        self.window.destroyed.connect(self.stop)                                        # Ensuring threads are stopped when window is closed

    # Registration 
    '''
    UserDetail: Validates user details and moves to capture page
    Reg_Capture: Captures registration images and saves to disk
    EmbeddingGenerator: Generates embedding from captured images and saves to disk
    save_embedding: Saves generated embedding to disk and reloads matcher
    '''
    def UserDetail(self):                                                               
        self.person_name = self.name_input.text().strip().capitalize()
        if not self.person_name or not self.id_input.text().strip():   # Validating that both name and ID are entered before proceeding to capture page to ensure we have necessary details for registration
            self.styled_msg(self.window, "Warning", "Enter Name & ID")
            return
        self.stack.setCurrentIndex(2)

    def Reg_Capture(self):
        if self.last_frame is None:
            return
        os.makedirs("Data/Registration", exist_ok=True)                                                 # Ensure registration directory exists                  
        cv2.imwrite(f"Data/Registration/{self.person_name}_{self.capture_score}.jpg", self.last_frame)  # Save captured image to registration directory with unique name
        self.capture_counter_reg.setText(f"{self.capture_score}")                                       # Showing capture counter on UI
        self.capture_score += 1                                                                         # Incrementing capture score to keep track of number of images captured

    def EmbeddingGenerator(self):
        self.emb_worker = EmbeddingWorker("Data/Registration")  # Initialize embedding worker with path to registration images
        self.emb_worker.finished.connect(self.save_embedding)   # Connect the finished signal of embedding worker to save_embedding function to save the generated embedding once it's done
        self.emb_worker.start()                                 # Start the embedding worker thread to generate embedding in background without freezing the UI

    def save_embedding(self, emb):
        if self.capture_score > 80:                                  # Ensure that at least 80 images were captured to generate a reliable embedding
            person_dir = f"Data/embeddings/{self.person_name}"       # Create a directory for the person inside embeddings directory to save their embedding
            os.makedirs(person_dir, exist_ok=True)                   # Ensure that the person's embedding directory exists   

            np.save(f"{person_dir}/{self.person_name}_embedding.npy",emb.astype(np.float32))  # Save the generated embedding as a .npy file in the person's embedding directory

            self.styled_msg(self.window, "Success", "Embedding saved")
            
            shutil.rmtree("Data/Registration", ignore_errors=True)   # Clean up registration images after embedding is saved to free up space and ensure fresh captures for next registration

            # reload matcher
            self.matcher.reload()                                    # Reload the matcher to include the new embedding without needing to restart the application
            self.stack.setCurrentIndex(0)
        else:
            self.styled_msg(self.window, "Fail", "Capture Minimum 80 Images")    # Show a message box if the user tries to save an embedding without capturing at least 80 images to ensure they understand the requirement for a reliable embedding

    # ---------- Inference ----------
    def update_frame_buffer(self, frame):
        
        if not self.inference_started:
            self.inf_worker.start()
            self.inference_started = True
        self.inf_worker.update_frame(frame)
        self.ImageUpdateSlot(frame)


    def update_bbox(self, bbox):         # Update the current bounding box coordinates received from the inference thread to be drawn on the camera feed in the UI
        if bbox == None:
            self.current_bbox = [0, 0, 0, 0]
        else:
            self.current_bbox = bbox

    def update_result(self, sim, name):     # Update the current similarity score and detected name received from the inference thread to be displayed on the camera feed in the UI
        self.current_sim = sim           
        self.current_name = name
        if sim is None or sim < 0.35:                # If similarity score is below threshold, consider it as unknown to avoid false positives
            self.detected_name.setText("Unknown")
        else:
            self.detected_name.setText(f"{name}")

    # ---------- Display ----------
    def ImageUpdateSlot(self,frame):
        if frame is None:
            return "No Frame"

        if self.current_bbox:                       
            x1, y1, x2, y2 = self.current_bbox 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.current_sim and self.current_sim > 0.35:
                text = f"{self.current_name} ({self.current_sim:.2f})"
            else:
                text = "Unknown"

            cv2.putText(
                frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qt_img)
        
        if self.stack.currentIndex() == 0 :
            label = self.camera_label
        else:
            label = self.camera_label_reg
            
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def capture_Login(self):
        date_str = datetime.now().strftime("%d-%m-%Y")
        self.styled_msg(self.window, "Success", f"{self.current_name}\n{date_str}")
        

    def stop(self):
        self.worker1.stop()
        self.inf_worker.stop()
    

    def styled_msg(self ,parent, title, text, bg="#1e1e1e", fg="#00ff99"):
        msg = QMessageBox(parent)
        msg.setWindowTitle(title)
        msg.setText(text)

        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {bg};
            }}
            QLabel {{
                color: {fg};
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                padding: 6px 14px;
            }}
        """)

        msg.exec()


# ================= THREADS =================

class InferenceWorker(QThread):
    result = Signal(float, str)
    bboxs = Signal(list)

    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.frame = None
        self.running = True

    def update_frame(self, frame):
        self.frame = frame

    def run(self):
        while self.running:
            if self.frame is None:
                self.msleep(5)
                continue

            sim, name, bbox = self.matcher.match(self.frame)
            if bbox is not None:
                self.result.emit(sim, name)
                self.bboxs.emit(bbox)

            self.msleep(30)

    def stop(self): # Stop the inference thread by setting running to False and waiting for it to finish before exiting to ensure clean shutdown
        self.running = False
        self.wait()


class EmbeddingWorker(QThread):    
    finished = Signal(np.ndarray)    # Signal to emit the generated embedding back to main thread

    def __init__(self, path):        # Initialize with path to registration images
        super().__init__()
        self.path = path             # Storing path for use in run method

    def run(self):
        emb = Embedding(self.path)   # Generate embedding using the Embedding function from Embedding.py
        self.finished.emit(emb)      # Emit the generated embedding back to main thread using the finished signal


class CameraThread(QThread):         
    Img_npArray = Signal(object)    # Signal to emit the captured frame as a numpy array for use in inference

    def run(self):
        self.active = True
        cap = cv2.VideoCapture(0)
        # Setting a lower resolution for faster processing while still maintaining enough detail for face recognition
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.active:
            ret, frame = cap.read()
            if not ret:
                continue

            self.Img_npArray.emit(frame)                                          # Emit the original frame as a numpy array for use in inference to detect faces and generate embeddings
            self.msleep(10)                                                       # Sleep for a short duration to control the frame rate and reduce CPU usage                      

        cap.release()

    def stop(self):                                                               # Stop the camera thread by setting active to False  
        self.active = False
        self.wait()                                                               # Wait for the thread to finish before exiting to ensure clean shutdown


# ================= MAIN =================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AppWindow()
    win.window.show()
    sys.exit(app.exec())
