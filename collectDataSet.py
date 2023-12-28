import cv2
import os
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time


TIME_INTERVAL = 500  
GESTURE_NAME = "A"
ITERATOR = 0

class ImageCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Capture App")

        self.label = ttk.Label(root)
        self.label.grid(row=0, column=0, columnspan=2)

        self.cap = cv2.VideoCapture(0)
        self.capture_thread = threading.Thread(target=self.update_video_feed, daemon=True)
        self.capture_thread.start()

        self.root.after(TIME_INTERVAL, lambda:self.capture_image(ITERATOR))

    def update_video_feed(self):
        while True:
            ret, frame = self.cap.read()
            # display horizontal flip of frame
            if ret:
                cv2.rectangle(frame, (100, 100), (302, 302), (0, 255, 0), 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.label.configure(image=photo)
                self.label.image = photo

    def capture_image(self, iter):
        create_directory(GESTURE_NAME)
        image_name = os.path.join(GESTURE_NAME, f"{iter}.jpg")
        ret, frame = self.cap.read()
        if ret:
            frame = frame[100:300, 100:300, :]
            cv2.imwrite(image_name, frame)
            print(f"Image captured and saved as {image_name}")
        self.root.after(TIME_INTERVAL, lambda:self.capture_image(iter+1))

if __name__ == "__main__":
    def create_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    root = tk.Tk()
    app = ImageCaptureApp(root)
    app.root.mainloop()
