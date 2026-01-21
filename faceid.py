# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np
from keras.models import load_model

# Build app and layout 
class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = load_model('siamese_model_best.h5', custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})
        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img

    # Verification function to verify person
    def verify(self, *args):
            # Specify thresholds
            detection_threshold = 0.4
            verification_threshold = 0.2
            
            # 1. 捕捉當前畫面並存檔
            SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
            # 捕捉畫面後，存檔前先轉色
            ret, frame = self.capture.read()
            frame = frame[120:120+250, 200:200+250, :]
            # 加入這一行！
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            cv2.imwrite(SAVE_PATH, rgb_frame)

            # 【修正處 1】：讀取並預處理剛剛存下來的這張 input_image.jpg
            input_img = self.preprocess(SAVE_PATH)
            
            # 2. 擴充維度變成 (1, 100, 100, 3)
            input_img = np.expand_dims(input_img, axis=0)

            # Build results array
            results = []
            
            # 3. 遍歷驗證資料夾中的所有圖片
            verify_path = os.path.join('application_data', 'verification_images')
            for image in os.listdir(verify_path):
                # 預處理驗證圖片
                validation_img = self.preprocess(os.path.join(verify_path, image))
                validation_img = np.expand_dims(validation_img, axis=0)
                
                # 模型預測
                result = self.model.predict([input_img, validation_img])
                results.append(result)
                test_self = self.model.predict([input_img, input_img])
                print(f"DEBUG - 自己比對自己得分: {test_self}")
        
            # Detection Threshold: Metric above which a prediciton is considered positive 
            detection = np.sum(np.array(results) > detection_threshold)
            
            # Verification Threshold: Proportion of positive predictions / total positive samples 
            verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
            verified = verification > verification_threshold

            # Set verification text 
            self.verification_label.text = 'Verified' if verified == True else 'Unverified'

            # Log out details
            Logger.info(results)
            Logger.info(detection)
            Logger.info(verification)
            Logger.info(verified)

            
            return results, verified



if __name__ == '__main__':
    CamApp().run()