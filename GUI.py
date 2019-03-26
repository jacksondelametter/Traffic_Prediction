import os
from tkinter import *
import matplotlib.image as image
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import sys
from random import sample
from keras.models import model_from_json
import keras
import numpy as np
from keras.preprocessing import image

os.chdir('..')
current_dir = os.getcwd()
pic_dir = os.path.join(current_dir, 'test')
json_path = os.path.join(current_dir, 'model')
json_file = open('model.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')
print('Loaded model with weights')

class main:
    def __init__(self, master):
        
        self.master = master
        self.get_pictures()
        self.picture_frame = Frame(self.master, padx=5, pady=5)
        self.picture_label = Label(self.picture_frame)
        
        self.picture_label.pack()
        self.picture_frame.pack(side=LEFT)
        info_frame = Frame(self.master, padx=5, pady=5)
        Label(info_frame,text="Traffic Predicting",fg="black",font=("",20,"bold")).pack(pady=10)
        self.prediction_label = Label(info_frame,text="Traffic Prediction: None",fg="blue",font=("",20,"bold"))
        self.answer_label = Label(info_frame,text="Traffic Answer: None",fg="blue",font=("",20,"bold"))
        self.prediction_label.pack(pady=20)
        self.answer_label.pack(pady=20)
        self.next_picture()
        
        arrow_frame = Frame(info_frame, pady=20)
        self.next_button = Button(arrow_frame,font=("",10),fg="white",bg="red", text="Next", command=self.next_picture)
        self.prev_button = Button(arrow_frame,font=("",10),fg="white",bg="red", text="Prev", command=self.prev_picture)
        self.next_button.pack(side=RIGHT)
        self.prev_button.pack(side=LEFT)
        arrow_frame.pack(side=BOTTOM)

        Button(info_frame,font=("",15),fg="white",bg="red", text="Predict", command=self.predict_traffic).pack(side=BOTTOM)
        info_frame.pack(side=RIGHT,fill=Y)


    def predict_traffic(self):
        print('Predict traffic')
        label = self.picture_dic[self.current_pic]
        pic_path = os.path.join(pic_dir, label)
        pic_path = os.path.join(pic_path, self.current_pic)
        pic = image.load_img(pic_path, target_size=(150, 150))
        pic_array = image.img_to_array(pic)
        pic_array = pic_array / 255
        img = np.expand_dims(pic_array, axis=0)
        print(img.shape)
        result = model.predict_classes(img)
        prediction = result[0]
        if prediction == 0:
            prediction = 'low'
        else:
            prediction = 'medium'
        print(prediction)
        self.prediction_label['text'] = 'Traffic Prediction: {}'.format(prediction)
        self.answer_label['text'] = 'Traffic Answer: ' + label

    def next_picture(self):
        print('Next picture')
        if self.picture_index < len(self.picture_list) - 1:
            self.picture_index += 1
            self.prediction_label['text'] = "Traffic Prediction: None"
            self.answer_label['text'] = "Traffic Answer: None"
        img = self.get_next_picture()
        self.picture_label.configure(image=img)
        self.picture_label.image = img

    def prev_picture(self):
        print('Prev picture')
        if self.picture_index > 0:
            self.picture_index -= 1
            self.prediction_label['text'] = "Traffic Prediction: None"
            self.answer_label['text'] = "Traffic Answer: None"
        img = self.get_next_picture()
        self.picture_label.configure(image=img)
        self.picture_label.image = img


    def get_next_picture(self):
        print('Getting pictures')
        self.current_pic = self.picture_list[self.picture_index]
        picture_label = self.picture_dic[self.current_pic]
        picture_path = os.path.join(pic_dir, picture_label)
        picture_path = os.path.join(picture_path, self.current_pic)
        img = ImageTk.PhotoImage(file=picture_path)
        return img

    def get_pictures(self):
        low_dir = os.path.join(pic_dir, 'low')
        medium_dir = os.path.join(pic_dir, 'medium')
        pics = {}
        low_pics = self.get_pictures_from_dir(low_dir, pics, 'low')
        medium_pics = self.get_pictures_from_dir(medium_dir, pics, 'medium')
        self.picture_dic = pics
        self.picture_list = sample(pics.keys(), len(pics.keys()))
        self.picture_index = -1

    def get_pictures_from_dir(self, dir, pics, label):
        count = 0
        for pic_name in os.listdir(dir):
            if count != 100:
                pics[pic_name] = label

root = Tk()
main(root)
root.title('Traffic detector')
root.resizable(0, 0)
root.mainloop()
