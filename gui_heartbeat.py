#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tkinter

from tkinter import *
from tkinter.filedialog import askopenfilename
import classify as c
import pygame
from pygame import mixer
from keras.models import model_from_json

model_json='hb_model_orthogonal_experiment_norm_vgg16_adam.json'
weights='hb_weights_orthogonal_experiment_norm_vgg16_adam.hdf5'

# Loads model from Json file
model = model_from_json(open(model_json).read())
# Loads pre-trained weights for the model
model.load_weights(weights)
# Compiles the model
model.compile(loss='categorical_crossentropy', optimizer='sgd')


pygame.init()
mixer.init()

top = Tk()
top.title = 'Normal/Abnormal Heartbeat'
top.geometry('300x170')

T = Text(top, height=2, width=20)
T.grid(row=1, column=0, columnspan=2)
T.configure(state='disabled')

label_name = ['Abnormal', 'Normal']
t1 = Text(top, bd=0, width=20, height=2, font='Fixdsys -14')
t1.grid(row=5, column=0, columnspan=2)
t1.configure(state='disabled')


def showAudioFile():
    t1.configure(state='normal')
    t1.delete(0.0, tkinter.END)
    t1.update()
    t1.configure(state='disabled')

    global file
    file = askopenfilename(title='Open Image')
    T.configure(state='normal')
    T.delete(0.0, tkinter.END)
    file_name = file[file.rfind("/") + 1:]
    T.insert('insert', file_name)
    T.tag_add('highlight1', '1.0', '20.0')
    T.tag_configure('highlight1', justify="center")
    T.update()
    T.configure(state='disabled')
    mixer.music.load(file)
    mixer.music.play(-1)


add_button = Button(top, text='Add', command=showAudioFile)
add_button.grid(row=0, column=0, columnspan=2)


def stopAudioFile():
    mixer.music.pause()


def continueAudioFile():
    mixer.music.unpause()


pause_button = Button(top, text='Pause', command=stopAudioFile)
pause_button.grid(row=2, column=0)

continue_button = Button(top, text='Continue', command=continueAudioFile)
continue_button.grid(row=2, column=1)


def classify():
    print(c.test(file, model))
    if c.test(file, model) == 0:
        text = label_name[1]
        bgcolor = 'green'
    else:
        text = label_name[0]
        bgcolor = 'red'
    t1.configure(state='normal')
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', text + '\n')
    t1.tag_add('highlight', '1.0', '20.0')
    t1.tag_configure('highlight', justify="center", background=bgcolor)
    t1.update()
    t1.configure(state='disabled')


classify_button = Button(top, text='Classify', command=classify)
classify_button.grid(row=4, column=0, columnspan=2)

l1 = Label(top, text='Please <Add> an audio file (.wav), then press <Classify> ')
l1.grid(row=6, columnspan=2)

top.mainloop()
