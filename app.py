import cv2
import threading
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from keras.models import load_model
from keras.models import model_from_json

THRESHOLD = 60
json_file = open("signlanguagedetectionmodel48x48.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagedetectionmodel48x48.h5")


class App():
    root = Tk()
    def __init__(self):
        root = self.root
        root.title("Image Capture App")
        root.resizable(False, False)
        root.state("zoomed")
        root.configure(bg="whitesmoke")
        
        self.active = 0
        self.eraseimg = Image.open("images/erase.png")
        resized = self.eraseimg.resize((30, 30))
        self.eraseimg = ImageTk.PhotoImage(resized)
        
        root.option_add("*Font", "Roboto")
        
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        
        mainframe = Frame(root, bg="#806ce8")
        mainframe.configure(width=self.screen_width, height=40)
        mainframe.pack()
        mainframe.pack_propagate(0)
        
        leftframe = Frame(root, bg="#17152d")
        leftframe.configure(width=(3*self.screen_width)//5, height=self.screen_height)
        leftframe.pack(side=LEFT)
        leftframe.pack_propagate(0)
        
        rightframe = Frame(root, bg="#2d275c")
        rightframe.configure(width=(2*self.screen_width)//5, height=self.screen_height)
        rightframe.pack(side=RIGHT)
        rightframe.pack_propagate(0)
  
        Label(mainframe, text="Sign Language Translator", bg="#806ce8", fg="white",
             font=('Roboto',15)).pack(pady=5)
    
        self.renderRight(rightframe)
        self.renderLeft(leftframe)
        
    def run(self):
        self.root.mainloop()
        
    def enter(self, e):
        e.widget.config(cursor = "hand2")
    def leave(self, e):
        e.widget.config(cursor = "arrow")
        
    def renderRight(self, frame):
        
        fr = Frame(frame, bg="#2d275c")
        fr.pack(pady=15)
        start = Button(fr, text="Start Predicting", bg="#5733d5", fg="white",
               relief=FLAT, height=2, width=17, command=self.startPredicting,
                font=('Roboto',13))
        start.pack(side="left", pady=10, padx=15)
        
        stop = Button(fr, text="Stop Predicting", bg="#5733d5", fg="white",
               relief=FLAT, height=2, width=17, command=self.stopPredicting,
                font=('Roboto',13))
        stop.pack(side="left",pady=10)
        
        self.current_text = Label(frame, bg="#2d275c", fg="white",
                font=('Roboto',250, 'bold'))
        self.current_text.pack(pady=40)
        
        start.bind("<Enter>", self.enter)
        start.bind("<Leave>", self.leave)
        stop.bind("<Enter>", self.enter)
        stop.bind("<Leave>", self.leave)
        
        
    def renderLeft(self, frame):
        Label(frame, text="Camera", bg="#17152d", fg="white",
             font=('Roboto',12, 'bold')).pack(pady=15, padx=70, anchor='w')
        
        width = (3 * self.screen_width)//5 - 150
        height = self.screen_height//2
                 
        canvas = Label(frame, bg="#17152d")
        canvas.configure(width=width, height=height)
        canvas.pack()
        
        self.cap = cv2.VideoCapture(0)
        t1 = threading.Thread(target=self.runCamera, args=(canvas, 0))
        t1.start()
        
        fr = Frame(frame, bg="#17152d")
        fr.pack()
        
        self.entrybox = Entry(fr, bg="#292369", fg="white", font=('Roboto',20),
                              width=37, insertbackground="white",
                                relief=FLAT)
        self.entrybox.pack(pady=50, padx=(65,10), anchor='w', side="left")
        
        button = Label(fr, image=self.eraseimg, bg="#17152d", fg="white")
        button.pack(pady=50, padx=(10,70), anchor='e', side="left")
        
        button.bind("<Button-1>", lambda event: self.erase())
        button.bind("<Enter>", self.enter)
        button.bind("<Leave>", self.leave)
                       
                       
    def erase(self):
        self.entrybox.delete(len(self.entrybox.get())-1, END)
        
    def runCamera(self, label, limit):
        # draw bounding box
        width = (3 * self.screen_width)//5 - 150
        height = self.screen_height//2
        
        x = width//2 - 90
        y = height//2 - 30
        
        cv2image= cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        image = cv2image[y:y+200, x:x+200]
        cv2.imwrite("capture.jpg", image)
        
        limit = limit + 1
        
        # image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
        if limit == THRESHOLD:
            limit = 0
            self.send_to_predictor()
        
        cv2.rectangle(cv2image, (x, y), (x+200, y+200), (43, 121, 255), 2)

        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image = img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(20, lambda: self.runCamera(label, limit))
        
    def send_to_predictor(self):
        if self.active == 1:
            # img = np.array(img)
            label = self.predict()
            
            if label == "Space":
                self.entrybox.insert(END, " ")
                self.current_text.config(text="_")
            elif label == "Blank":
                self.current_text.config(text="")
                
            else:
                self.entrybox.insert(END, label)
                self.current_text.config(text=label)
            
            if len(self.entrybox.get()) > 30:
                self.entrybox.delete(0, END)
            
        else:
            self.current_text.config(text="")
            
    def startPredicting(self):
        self.active = 1
    def stopPredicting(self):
        self.active = 0
        
    def extract_features(self, image):
        feature = np.array(image)
        feature = feature.reshape(1,48,48,1)
        return feature/255.0
    
    def predict(self):
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 
         'G', 'H', 'I', 'J', 'K', 'L', 'M', 
         'N', 'O', 'P', 'Q', 'R', 'S', 'Space',
         'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        frame = cv2.imread("capture.jpg")
        
        cropframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cropframe = cv2.resize(cropframe,(48,48))
        cropframe = self.extract_features(cropframe)
        pred = model.predict(cropframe)
        print(pred)
        
        prediction_label = labels[pred.argmax()]
        if pred[0][pred.argmax()] < 0.4:
            prediction_label = "Blank"
        return prediction_label
        
    
app = App()
app.run()
