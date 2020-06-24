'''
GUI utility for the prediction display purpose.

Uses Tkinter library and shows the letters predicted and gives ability to 
either use the predicted letter into formation of words and sentences or deleting 
the letters.
'''

import tkinter
from tkinter import *
import PIL.Image, PIL.ImageTk
import cv2
import numpy as np
from keras.preprocessing import image
import operator
from collections import Counter
import tkinter.font as font

from keras.models import load_model
classifier = load_model('Trained_model.h5')

def predictor():
	test_image = image.load_img('1.png', target_size=(64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)

	prediction = {'A': result[0][0], 'B': result[0][1], 'C': result[0][2],
				  'D': result[0][3], 'E': result[0][4], 'F': result[0][5], 
				  'G': result[0][6], 'H': result[0][7], 'I': result[0][8], 
				  'J': result[0][9], 'K': result[0][10], 'L': result[0][11], 
				  'M': result[0][12], 'N': result[0][13], 'O': result[0][14], 
				  'P': result[0][15], 'Q': result[0][16], 'R': result[0][17], 
				  'S': result[0][18], 'T': result[0][19], 'U': result[0][20], 
				  'V': result[0][21], 'W': result[0][22], 'X': result[0][23], 
				  'Y': result[0][24], 'Z': result[0][25]
				  }

	predicted = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
	# print(predicted[0])
	if(predicted[0][1] > 0.98):
		return predicted[0][0]
	else:
		return ''



root = tkinter.Tk()


class App:
	def __init__(self, window, video_source=0):
		self.img_text = ''
		self.variable_letter = StringVar()
		self.variable_text = StringVar()

		self.window = window
		self.window.title("Gesture Recognition")
		self.video_source = video_source

		self.vid = vid_cap(self.video_source)

################## IMAGE FRAME ##################
		self.col0_frame = tkinter.Frame(window)
		self.col0_frame.grid(row=0,column=0)
		self.image_frame = tkinter.Canvas(self.col0_frame, height=480, width=640)
		self.image_frame.grid(row=0, column=0)
################## IMAGE FRAME ##################

################## LABEL FRAME ##################
		self.label_frame = tkinter.Frame(self.col0_frame)
		self.label_frame.grid(row=1, column=0)
		fontstyle = font.Font(family='Lucida Grande', size=22)

		self.pred_text_label = tkinter.Label(self.label_frame, text='Predicted Letter is: ', height=2, font=fontstyle)
		self.pred_text_label.grid(row=0, column=0)
		self.pred_text_label = tkinter.Label(self.label_frame, text='Predicted Text is: ', height=2, font=fontstyle)
		self.pred_text_label.grid(row=1, column=0)

		self.pred_letter_label = tkinter.Label(self.label_frame, textvariable=self.variable_letter, height=2, font=fontstyle)
		self.pred_letter_label.grid(row=0, column=1)
		self.pred_text_label = tkinter.Label(self.label_frame, textvariable=self.variable_text, height=2, font=fontstyle)
		self.pred_text_label.grid(row=1, column=1)
		self.enter_letter_button = tkinter.Button(self.label_frame, text='Enter Letter into Text')
		self.enter_letter_button.grid(row=1, column=2)
		self.window.bind('<Return>', self.enter_letter_into_text)
		self.window.bind('<BackSpace>', self.delete_letter_from_text)
		self.window.bind('<space>', self.enter_space_into_text)
		self.enter_letter_button.bind('<Button-1>', self.enter_letter_into_text)
################## LABEL FRAME ##################

################## MASK FRAME ##################
		self.col1_frame = tkinter.Frame(window)
		self.col1_frame.grid(row=0,column=1, sticky='nw')
		self.mask_frame = tkinter.Canvas(self.col1_frame, height=190, width=190)
		self.mask_frame.grid(row=0, column=1, padx=10, pady=5)
		fontstyle_1 = font.Font(family='Lucida Grande', size=10)
		self.scrollbar_desc = tkinter.Label(self.col1_frame, text="Scrollbars to adjust HSV ranges", font=fontstyle_1, bd=3)
		self.scrollbar_desc.grid(row=1, column=1)		
		self.slider_frame = tkinter.Frame(self.col1_frame)
		self.slider_frame.grid(row=2, column=1, padx=10, pady=5)
################## MASK FRAME ##################


################## CHART FRAME ##################
		self.chart_button = tkinter.Button(self.col1_frame, text='Show Sign Language Chart', command=self.create_chart_window, padx=10, pady=10)
		self.chart_button.grid(row=3, column=1)
################## CHART FRAME ##################

################## SLIDER FRAME ##################
		self.slider_H_Low = tkinter.Scale(self.slider_frame, from_=0, to=179, orient='horizontal')
		self.slider_H_Low.set(0)
		self.slider_H_Low.grid(row=0, column=1, padx=10, pady=2)
		self.lbl_H_Low = Label(self.slider_frame, text='H LOW')
		self.lbl_H_Low.grid(row=0, column=0, padx=10, pady=5)

		self.slider_S_Low = tkinter.Scale(self.slider_frame, from_=0, to=255, orient='horizontal')
		self.slider_S_Low.set(0)
		self.slider_S_Low.grid(row=1, column=1, padx=10, pady=2)
		self.lbl_S_Low = Label(self.slider_frame, text='S LOW')
		self.lbl_S_Low.grid(row=1, column=0, padx=10, pady=5)

		self.slider_V_Low = tkinter.Scale(self.slider_frame, from_=0, to=255, orient='horizontal')
		self.slider_V_Low.set(0)
		self.slider_V_Low.grid(row=2, column=1, padx=10, pady=2)
		self.lbl_V_Low = Label(self.slider_frame, text='V LOW')
		self.lbl_V_Low.grid(row=2, column=0, padx=10, pady=5)


		self.slider_H_High = tkinter.Scale(self.slider_frame, from_=0, to=179, orient='horizontal')
		self.slider_H_High.set(179)
		self.slider_H_High.grid(row=0, column=3, padx=10, pady=2)
		self.lbl_H_High = Label(self.slider_frame, text='H HIGH')
		self.lbl_H_High.grid(row=0, column=2, padx=10, pady=5)

		self.slider_S_High = tkinter.Scale(self.slider_frame, from_=0, to=255, orient='horizontal')
		self.slider_S_High.set(255)
		self.slider_S_High.grid(row=1, column=3, padx=10, pady=2)
		self.lbl_S_High = Label(self.slider_frame, text='S HIGH')
		self.lbl_S_High.grid(row=1, column=2, padx=10, pady=5)

		self.slider_V_High = tkinter.Scale(self.slider_frame, from_=0, to=255, orient='horizontal')
		self.slider_V_High.set(255)
		self.slider_V_High.grid(row=2, column=3, padx=10, pady=2)
		self.lbl_V_High = Label(self.slider_frame, text='V HIGH')
		self.lbl_V_High.grid(row=2, column=2, padx=10, pady=5)
################## SLIDER FRAME ##################

		self.delay = 15
		self.update()

		self.window.mainloop()

################## FUNCTIONS #####################
	def create_chart_window(self):
		self.window_chart = tkinter.Toplevel(root)
		self.window_chart.title('Sign Language Chart')
		self.chart_frame = tkinter.Canvas(self.window_chart, height=368, width=480)
		self.img_chart = cv2.imread(r"C:\Users\Aayush.Ayush_PC\PycharmProjects\HandGestureRecognition\plots\slchart.jpg")
		# self.img_chart = cv2.resize(self.img_chart, (250, 350))
		self.chart = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.img_chart))
		self.chart_frame.create_image(0, 0, image=self.chart, anchor='nw')
		self.chart_frame.grid(row=0, column=0)


	def enter_letter_into_text(self,event):
		self.variable_text.set(self.variable_text.get() + self.variable_letter.get())

	def delete_letter_from_text(self, event):
		self.variable_text.set(self.variable_text.get()[0:-1])

	def	enter_space_into_text(self, event):
		self.variable_text.set(self.variable_text.get() + ' ')

	def update(self):
		ret, frame = self.vid.get_frame()
		if ret:
			self.l_h = self.slider_H_Low.get()
			self.l_s = self.slider_S_Low.get()
			self.l_v = self.slider_V_Low.get()
			self.u_h = self.slider_H_High.get()
			self.u_s = self.slider_S_High.get()
			self.u_v = self.slider_V_High.get()

			self.img = cv2.rectangle(frame, (425,100), (625,300), (0,255,0), thickness=2, lineType=8, shift=0)

			self.lower_blue = np.array([self.l_h, self.l_s, self.l_v])
			self.upper_blue = np.array([self.u_h, self.u_s, self.u_v])		
			self.imcrop = self.img[102:298, 427:623]
			self.hsv = cv2.cvtColor(self.imcrop, cv2.COLOR_BGR2HSV)
			#self.hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
			self.mask = cv2.inRange(self.hsv, self.lower_blue, self.upper_blue)

			self.img_from_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
			self.image_frame.create_image(0, 0, image=self.img_from_frame, anchor="nw")
			#print(self.img_from_frame.height(), self.img_from_frame.width())

			self.img_mask = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.mask))
			self.mask_frame.create_image(0, 0, image=self.img_mask, anchor="nw")
			# cv2.imshow('frame', frame)
			# cv2.imshow('mask', self.mask)

			#putText(frame, self.img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))

			img_name = '1.png'
			save_img = cv2.resize(self.mask, (64, 64))
			cv2.imwrite(img_name, save_img)
			counter_list = []
			i=0
			while i < 20:
				_a = predictor()
				if i % 2 ==  0:
					counter_list.append(_a)
					# print(_a)
				i+=1
			count = Counter(counter_list)
			# print(counter_list)
			counter_list.clear()
			# print(counter_list)
			most_common = count.most_common(1)[0][0]
			self.variable_letter.set(most_common)

			self.window.after(self.delay, self.update)
################## FUNCTIONS #####################


class vid_cap:
	def __init__(self, video_source=0):
		self.vid = cv2.VideoCapture(video_source)
		if not self.vid.isOpened():
			raise ValueError('Unable to open Video Source', video_source)


	def get_frame(self):
		if self.vid.isOpened():
			ret, frame = self.vid.read()
			if ret:
				return ret, cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 1)
			else:
				return ret, None
		else:
			return ret, None


	def __del__(self):
		if self.vid.isOpened():
			self.vid.release()


App(root)
