from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import object_detection as od
import imageio
import cv2

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, mater)

        self.master
        self.pos = []
        self.line = []
        self.rect = []
        self.mater.title("Traffic detection")
        self.pack(fill=BOTH, expand=1)

        self.counter = 0

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add.command
