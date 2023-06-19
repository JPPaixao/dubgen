"""
Created on Wed Aug  3 13:36:40 2022

@author: João Paixão
@email: joao.p.paixao@tecnico.ulisboa.pt

@brief: Graphical User Interface
"""


import tkinter as tk               
from tkinter import font as tkfont  
import threading

from PIL import ImageTk, Image

import os
import sys

#so that pygame doesn't print anything
devnull = open(os.devnull, 'w')
sys.stdout = devnull
import pygame
sys.stdout = sys.__stdout__

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import warnings

class App(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.shared_data = {
           "main_path": os.getcwd().replace('\\','/'),
           "bpm": tk.IntVar(),
           'train' : tk.BooleanVar(),
           'mod': tk.DoubleVar(),
           'fb': tk.IntVar(),
           'section_size': tk.IntVar()}

        
        self.title_font = tkfont.Font(family='Helvetica', size=38,
                                      weight="bold", slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        #container.grid(row=0, column=3, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(2, weight=2)


        self.frames = {}
        for F in (StartPage, PageOne, PageOne2, PageTwo, PageThree, PageFour,
                  PageFive, PageSix, PageSeven, PageEight, PageNine, PageTen,
                  
                  INFO_MOD_LEVEL, INFO_SECTION_SIZE, INFO_MOD_FEEDBACK,
                  INFO_CUSTOMIZE_PARAMETERS, INFO_CUTOFF, INFO_LFO, 
                  INFO_LFO_SHAPE_EVO, INFO_GRAIN_SIZE, INFO_GRAIN_SPACE,
                  INFO_GRAIN_ORDER, INFO_SMOOTHING, INFO_EVEN_SPACING,
                  INFO_SIGNAL_SHAPE_EVO):
            
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=2, sticky="nsew")
            
            #frame.pack(side="top", fill="both", expand=True)
            #frame.grid_rowconfigure(0, weight=1)
            #frame.grid_columnconfigure(0, weight=1)

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        controller.title("dubgen")
        controller.geometry("700x500")
        controller.config(bg="blue") 
        
        label = tk.Label(self, text="dubgen", font=controller.title_font, fg="#1c4966")
        #label.pack(side="top", fill="x", pady=10)
        label.pack(fill="both", expand=True)
        
        slogan = tk.Label(self, text="Sampler and MIDI Arrangement Generator",
                          font=('Helvetica', 18))
        slogan.pack( expand=True)

        button1 = tk.Button(self, text="Next", font=('Helvetica', 18),
                            command=lambda: controller.show_frame("PageOne"))

        button1.pack( expand=True)


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        row=0

        label = tk.Label(self, text="How it works:", font=controller.title_font)
        label.pack(side="top", pady=10)
        row+=1
        
        text1="dubgen creates monophonic samples from given sounds, modifies MIDI compositions, and then choses a sequence of samples to generate an arrangement (new MIDI + samples)"
        text2 = 'Created samples can also be modulated by a Synthesizer' 

        explanation1 = tk.Label(self, text=text1,
                          font=('Helvetica', 18, 'bold'), wraplength=500)
        explanation1.pack(expand=True)
        row+=1
        
        
        explanation2 = tk.Label(self, text=text2,
                          font=('Helvetica', 18, 'bold'), wraplength=500)
        explanation2.pack(expand=True)
        row+=1


        button = tk.Button(self, text="Next", font=('Helvetica', 18),
                           command= lambda: controller.show_frame("PageOne2"))
        button.pack(expand=True, side='right', pady=20)
        row+=1

class PageOne2(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        row=0

        label = tk.Label(self, text="How it works:", font=controller.title_font)
        label.pack(side="top", pady=10)
        row+=1
        
        text1="Your input:"
        text1_1='(Go to ".../dubgen_APP/user" folder)'
        text2="1. Folders with sounds for three instruments (Bass, Harmony and Melody);\n"
        text2_2="2. Three MIDI files, one for each instrument."
        text3="dubgen's Output:"
        text4="1. Folder with created samples;\n"
        text4_2="2. Three new MIDI files, one for each instrument;\n"
        text4_3='3. Wav and mp3 files of the generated Arrangement (with or without Synthesizer Processing).\n'

        explanation1 = tk.Label(self, text=text1,
                          font=('Helvetica', 18, 'bold'))#, justify="left", wraplength=500)
        explanation1.pack(expand=True)
        row+=1
        
        explanation1_1 = tk.Label(self, text=text1_1,
                          font=('Helvetica', 10, 'bold',"italic"), fg="#1c4966")#, justify="left", wraplength=500)
        explanation1_1.pack(expand=True)
        row+=1


        text_box = tk.Text(self)
        text_box.insert('end', text2)
        text_box.insert('end', text2_2)
        text_box.pack(expand=True)
        text_box.configure(state='disabled', height=5, width=50)
        row+=1
        
        explanation2 = tk.Label(self, text=text3,
                          font=('Helvetica', 18, 'bold'))
        explanation2.pack(expand=True)
        row+=1

        text_box2 = tk.Text(self)
        text_box2.insert('end', text4)
        text_box2.insert('end', text4_2)
        text_box2.insert('end', text4_3)
        text_box2.pack(expand=True)
        text_box2.configure(state='disabled', height=5, width=50)
        row+=1

        button = tk.Button(self, text="Next", font=('Helvetica', 18),
                           command= lambda: controller.show_frame("PageTwo"))
        button.pack(expand=True, side='right', pady=20)
        row+=1
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command= lambda: controller.show_frame("PageOne"))
        button.pack(expand=True, side='right', pady=20, padx=20)
        row+=1
        
        
# funcs parameter will have the reference
# of all the functions that are 
# passed as arguments i.e "fun1" and "fun2"
def combine_funcs(*funcs):
  
    # this function will call the passed functions
    # with the arguments that are passed to the functions
    def inner_combined_func(*args, **kwargs):
        for f in funcs:
  
            # Calling functions with arguments, if any
            f(*args, **kwargs)
  
    # returning the reference of inner_combined_func
    # this reference will have the called result of all
    # the functions that are passed to the combined_funcs
    return inner_combined_func

class user_input:
    def __init__(self, bpm, main_path, mod, fb, section_size):
        self.bpm = bpm
        self.main_path = main_path
        self.mod = mod
        self.fb = fb
        self.section_size = section_size


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Create five columns
        for i in range(5):
            if i!=2:
                self.grid_columnconfigure(i, weight=1, uniform="equal")
        
        # Place widgets using grid()
        label1 = tk.Label(self, text=" ")
        label1.grid(row=0, column=0)
        
        label2 = tk.Label(self, text=" ")
        label2.grid(row=0, column=1)
        
        label3 = tk.Label(self, text=" ")
        label3.grid(row=0, column=2)
        
        label4 = tk.Label(self, text=" ")
        label4.grid(row=0, column=3)
        
        label5 = tk.Label(self, text=" ")
        label5.grid(row=0, column=4)
        
        row = 1
        
        
        label = tk.Label(self, text="Settings", font=controller.title_font)
        label.grid(row=row, column=2 , columnspan=1, sticky="nsew")
        row+=1
        
        label_bpm = tk.Label(self, text='BPM', font=('Helvetica', "12"), justify="left")
        label_bpm.grid(row=row, column=0, sticky = 'S')
        
        slider_length = 300
        
        self.controller.shared_data["bpm"].set(120)
        bpm_entry = tk.Scale(self, variable= self.controller.shared_data["bpm"],
                              from_= 80, to = 200, resolution=1, orient="horizontal",
                              length = slider_length)  
        bpm_entry.grid(row=row, column=2, columnspan=2)
        
        min_bpm = tk.Label(self, text='80', font=('Helvetica', "10"))
        min_bpm.grid(row=row, column=1, sticky = 'SE')
        
        max_bpm = tk.Label(self, text='120', font=('Helvetica', "10"))
        max_bpm.grid(row=row, column=4, sticky = 'SW')

        row+=1
        
        #################### #mod level #############################
        # Create a label for the slider
        mod_label = tk.Label(self, text="Mod Level:", font=('Helvetica', "12"), justify="left")
        mod_label.grid(row = row, column=0, padx=10)
        
        info1 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_MOD_LEVEL"))
        info1.grid(row=row, column=1, sticky = 'S')
        
        # Create a double variable to store the slider value
        mod_var = tk.DoubleVar()
        mod_var.set(0.7)  # Set an initial value for the slider
        
        min_mod = tk.Label(self, text='0.1', font=('Helvetica', "10"))
        min_mod.grid(row=row, column=1, sticky = 'SE')
        
        max_mod = tk.Label(self, text='1', font=('Helvetica', "10"))
        max_mod.grid(row=row, column=4, sticky = 'SW')
        
        # Create the slider widget
        mod_slider = tk.Scale(self, variable=mod_var, from_= 0.1, to = 1.0, resolution=0.01,
                              orient="horizontal", 
                              length = slider_length)
        mod_slider.grid(row = row, column=2, columnspan=2)
        self.controller.shared_data['mod']=mod_var
        row+=1
        
        #################### #mod feeback #############################
        fb_label = tk.Label(self, text="Mod Feedback:", font=('Helvetica', "12"), justify="left")
        fb_label.grid(row = row, column=0, padx=10, columnspan=2, sticky = 'W')
        
        info2 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_MOD_FEEDBACK"))
        info2.grid(row=row, column=1, sticky = 'S')
        
        # Create a double variable to store the slider value
        fb_var = tk.IntVar()
        fb_var.set(1)  # Set an initial value for the slider
        
        min_fb = tk.Label(self, text='0', font=('Helvetica', "10"))
        min_fb.grid(row=row, column=1, sticky = 'SE')
        
        max_fb = tk.Label(self, text='3', font=('Helvetica', "10"))
        max_fb.grid(row=row, column=4, sticky = 'SW')
        
        # Create the slider widget
        fb_slider = tk.Scale(self, variable=fb_var, from_= 0, to=3, orient="horizontal",
                             length=slider_length)
        fb_slider.grid(row = row, column=2, columnspan=2)
        self.controller.shared_data['fb']=fb_var
        
        row+=1
        #################### #section_size #############################
        sec_label = tk.Label(self, text="Section size\n (Number of bars):",
                             font=('Helvetica', "12"), justify="left")
        sec_label.grid(row = row, column=0, padx=10,columnspan=2, sticky = 'W')
        
        info3 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_SECTION_SIZE"))
        info3.grid(row=row, column=1, sticky = 'S')
        
        # Create a double variable to store the slider value
        sec_var = tk.IntVar()
        sec_var.set(8)  # Set an initial value for the slider
        
        min_sec = tk.Label(self, text='0', font=('Helvetica', "10"))
        min_sec.grid(row=row, column=1, sticky = 'SE')
        
        max_sec = tk.Label(self, text='16', font=('Helvetica', "10"))
        max_sec.grid(row=row, column=4, sticky = 'SW')
        
        # Create the slider widget
        sec_slider = tk.Scale(self, variable=sec_var, from_= 4, to=16, resolution=4,
                              orient="horizontal", length=slider_length)
        sec_slider.grid(row = row, column=2, columnspan=2)
        self.controller.shared_data['section_size']=sec_var
        
        row+=1
        
        ############################## TRAIN CLASSIFICATION #####################################
        row+=1
        train_label = tk.Label(self, text='Custom Instrument\n Classification',
                               font=('Helvetica', "10") , justify="left", wraplength=120)
        train_label.grid(row = row, column=0, pady=10, padx=10, columnspan=3, sticky = 'W')
        
        info = tk.Button(self, text="Info*",
                           command=lambda: controller.show_frame("PageThree"))
        info.grid(row=row, column=2, pady=10, sticky = 'W')
             
        # Create a boolean variable to store the checkbox value
        train = tk.BooleanVar()
        
        # Create the checkbox
        train_checkbox = tk.Checkbutton(self,
                                        variable=train, compound="right") #, anchor="w"
        self.controller.shared_data['train']=train
        
        # Position the checkbox and filename entry widgets in the GUI window
        train_checkbox.grid(row=row, column=1, columnspan=1, pady=10, sticky = 'nsew')
        row+=1
        
        button2 = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageOne"))
        
        button3 = tk.Button(self, text="Next", font=('Helvetica', 18),
                           command= lambda: controller.show_frame("PageFour"))
        
        #button1.grid(row=row, column=0, pady=10)
        button2.grid(row=row, column=0, columnspan=2, pady=30, padx=20, sticky='w')
        button3.grid(row=row, column=4, pady=30, padx=20)

        #to join the two functions in a button: 
            #combine_funcs(lambda: fun1(arguments), lambda: fun2(arguments))
            #https://www.geeksforgeeks.org/how-to-bind-multiple-commands-to-tkinter-button/
        

        
class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Train your own Model!", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text_train= 'Before picking samples for the final arrangement, we first have to classify all samples by Instrument Type.\n'
        text_train2 = 'You can train your own Instrument Classification Model by placing Samples in each Instrument category (BASS, HARMONY and MELODY) on the folders at ".../dubgen_APP/user/train_sounds"'
        
        train_info = tk.Label(self, text=text_train,
                              font=('Helvetica', 18), wraplength=500)
        train_info.pack(expand=True, padx=20)
        
        train_info2 = tk.Label(self, text=text_train2,
                              font=('Helvetica', 18), wraplength=500)
        train_info2.pack(expand=True, padx=20)
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageTwo"))
        button.pack(expand=True, padx=20)
        
    
class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        row=0
        
        label = tk.Label(self, text="Create Arrangement!", font=controller.title_font)     
        #label.grid(row=row, column=0 , columnspan=3, sticky="nsew", pady=100)
        row+=1       
        label.pack(side="top", fill="both", pady=10, expand=True)
        
        self.button1 = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageTwo"))
        
        self.t1=threading.Thread(target=self.run_sampling) #init thread
        
        self.button2 = tk.Button(self, text="Start", font=('Helvetica', 18),
                           command= combine_funcs(self.Threading,
                            lambda: self.button2.config(text="Creating...", state=tk.DISABLED, width=35)))
        
        self.button2.pack( pady=10, expand=True)
        row+=1
        self.button1.pack( pady=10, expand=True, side='left')
        
        #self.out_sampling = {'midi_list': list()}
        self.out_ga = {'music_path': 'init', 'best_score': float(0.0)}
        
    
    def Threading(self):
        # Call work function
        self.event = threading.Event()

        self.t1.start()
    
    def run_sampling(self): 
        path = self.controller.shared_data["main_path"]
        bpm = self.controller.shared_data["bpm"].get()
        mod = self.controller.shared_data["mod"].get()
        fb = self.controller.shared_data["fb"].get()
        section_size = self.controller.shared_data["section_size"].get()
        
        user_data = user_input(bpm, path, mod, fb, section_size)
        
        self.button1.configure(text = "Go Back", state= tk.DISABLED)

        from main import Sampling_and_Mod
        train_bool=self.controller.shared_data['train'].get()
        
        if train_bool==True:
            train=1
        else: train=0

        
        user_info, midi_info_out, sample_info, sample_train_info, section_info, midi_info_real = Sampling_and_Mod(user_data, self.event ,
                                                                                                                   train, self.button2)
        
        
        self.out_sampling = {'user_info': user_info, 'midi_info_out': midi_info_out,
                             'sample_info': sample_info, 'sample_train_info': sample_train_info,
                             'section_info': section_info, 'midi_info_real': midi_info_real}
        
        
        ######################################### GA PREPOP ###################################################
        
        self.in_pp = self.out_sampling #input genetic
        
        from main import main_GA_PREPOP
        info_GA, class_idx, Pop, df = main_GA_PREPOP(self.in_pp['user_info'],
                                    self.in_pp['midi_info_out'], self.in_pp['sample_info'], self.button2)
        
        
        self.out_ga_prepop = {'info_GA': info_GA , 'class_idx': class_idx , 'Pop': Pop , 'df': df}
        
        self.in_gen1 = self.out_ga_prepop #input genetic from prepop
        self.in_gen2 = self.out_sampling #input genetic from sampling

        
        ######################################### GA ###################################################
        from main import main_GA
        music_path, best_score, sample_Ind = main_GA(self.in_gen2['user_info'], self.in_gen2['sample_info'],
                                         self.in_gen1['class_idx'], self.in_gen2['midi_info_out'],
                                         self.in_gen1['Pop'], self.in_gen1['df'], self.in_gen1['info_GA'],
                                          self.in_gen2['midi_info_real'], self.button2)
        
        
        self.out_ga = {'music_path': music_path, 'best_score': best_score, 'sample_Ind': sample_Ind}
        #'''
        
        
        #reconfigure button to pass to next phase
        self.button2.configure(text = "Listen to Arrangement!", font=('Helvetica', 16),
                               command = lambda: self.controller.show_frame("PageFive"),
                               state=tk.NORMAL, width=25)
        

        
    def stop_thread(self): #stop the thread
        self.button3.configure(text = "Stopping Thread...", state= tk.DISABLED)
        
        if self.t1.is_alive():
            self.event.set()
            print('Stopping Thread...')
            self.t1.join() #wait for thread to end 
        
        self.button3.configure(text = "Stop", state= tk.NORMAL)


    def change_layout4(self):
        self.button2.configure(text = "Stop", command = self.stop_thread)

        
class PageFive(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Create five columns
        for i in range(7):
            self.grid_columnconfigure(i, weight=1, uniform="equal")
        
        # Give the center column a weight of 1
        #self.grid_columnconfigure(2, weight=1)
        
        # Place widgets using grid()
        label1 = tk.Label(self, text=" ")
        label1.grid(row=0, column=0)
        
        label2 = tk.Label(self, text=" ")
        label2.grid(row=0, column=1)
        
        label3 = tk.Label(self, text=" ")
        label3.grid(row=0, column=2)
        
        label4 = tk.Label(self, text=" ")
        label4.grid(row=0, column=3)
        
        label5 = tk.Label(self, text=" ")
        label5.grid(row=0, column=4)
        
        label6 = tk.Label(self, text=" ")
        label6.grid(row=0, column=5)
        
        label7 = tk.Label(self, text=" ")
        label7.grid(row=0, column=6)
        
        row=1
        
        label = tk.Label(self, text="Listen to the Arrangement", font=controller.title_font)
        label.grid(row=row, column=0, columnspan=7, pady=10)
        row+=1
        
        main_path = self.controller.shared_data["main_path"]
        
        #create playlist mixer
        self.mixer = pygame.mixer
        self.mixer.init()


        ###################################### icons ####################################
        #icon_path = main_path + '/GUI/icons/'
        icon_path = main_path + '/icons/'
        
        play_img = Image.open(icon_path + 'play.png')
        pause_img = Image.open( icon_path + 'pause.png')
        stop_img = Image.open( icon_path + 'stop.png')
        backward_img = Image.open( icon_path + 'backward.png')
        forward_img = Image.open( icon_path + 'forward.png')
        
        play_img = play_img.resize((20, 20))
        pause_img = pause_img.resize((20, 20))
        stop_img = stop_img.resize((20, 20))
        backward_img = backward_img.resize((20, 20))
        forward_img = forward_img.resize((20, 20))
        
        play_icon = ImageTk.PhotoImage(image = play_img, master = controller)
        pause_icon = ImageTk.PhotoImage(image = pause_img, master = controller)
        stop_icon = ImageTk.PhotoImage(image = stop_img, master = controller)
        backward_icon = ImageTk.PhotoImage(image = backward_img, master = controller)
        forward_icon = ImageTk.PhotoImage(image = forward_img, master = controller)
        
        ########################################## mp3 #######################################
        mp3_frame = tk.Frame(self, padx=10, pady=10,
                             highlightbackground="black", highlightthickness=1)
        mp3_frame.grid(row=row, column=1, columnspan=5, pady=10, sticky='nsew')
        
        for i in range(5):
            mp3_frame.grid_columnconfigure(i, weight=1, uniform="equal")
        
        pady_mp3 = 20
        
        self.paused = False
        self.save_start = 0
        
        play_btn = tk.Button(mp3_frame, image=play_icon, borderwidth=0, command= self.Play_Song)
        pause_btn = tk.Button(mp3_frame, image=pause_icon, borderwidth=0, command= self.Pause_Song)
        stop_btn = tk.Button(mp3_frame, image=stop_icon, borderwidth=0, command= self.Stop_Song)
        backward_btn = tk.Button(mp3_frame, image=backward_icon, borderwidth=0, command= self.Backward_Song)
        forward_btn = tk.Button(mp3_frame, image=forward_icon, borderwidth=0, command= self.Forward_Song)
        
        play_btn.image = play_icon
        pause_btn.image = pause_icon
        stop_btn.image = stop_icon
        backward_btn.image = backward_icon
        forward_btn.image = forward_icon
        
        backward_btn.grid(row = row, column = 0, pady=pady_mp3, sticky='nsew')
        stop_btn.grid(row = row, column = 1, pady=pady_mp3, sticky='nsew')
        play_btn.grid(row = row, column = 2, pady=pady_mp3, sticky='nsew')
        pause_btn.grid(row = row, column = 3, pady=pady_mp3, sticky='nsew')
        forward_btn.grid(row = row, column = 4, pady=pady_mp3, sticky='nsew')
        
        row+=1
        
        ######################################## volume #######################################
        
        def set_volume(val):
            volume = float(val) / 100
            self.mixer.music.set_volume(volume)
        
        label_vol = tk.Label(mp3_frame, text="Volume", font=('Helvetica', 16))
        label_vol.grid(row=row, column=0, columnspan=5, sticky='s')
        row+=1
        
        vol_slider = tk.Scale(mp3_frame, from_=0, to=100, orient='horizontal', command= set_volume,
                              length= 300)
        vol_slider.set(50)
        vol_slider.grid(row = row, column=1, columnspan=3)#, pady=10)
        row+=1
        
        ######################################################################################
        
        go_synth = tk.Button(self, text="Go to Synthesizer", font=('Helvetica', 18),
                            command=lambda: controller.show_frame("PageSix"))
        go_synth.grid(row = row, column=4, columnspan=2, pady=50, sticky='E')
        
        def close_window():
            self.controller.destroy()
        
        close_button = tk.Button(self, text="Close", font=('Helvetica', 18),
                                 command=close_window)
        close_button.grid(row = row, column=3, columnspan=1, pady=50)
        row+=1
        
    def Play_Song(self):
        self.results = self.controller.frames['PageFour'].out_ga
        song_path = self.results['music_path']
        #print('music path: ', song_path)
        
        self.mixer.music.load(song_path)
        self.mixer.music.play(loops=0)
    
    def Stop_Song(self):
        self.mixer.music.stop()
        
        self.start=0
        
    def Pause_Song(self):
        
        if self.paused:
            #unpause 
            self.mixer.music.unpause()
            self.paused = False
        else:
            #pause
            self.mixer.music.pause()
            self.paused = True
        
    def Forward_Song(self):
        
        start = self.save_start
        
        play_time = self.mixer.music.get_pos()

        start += (play_time/1000.0) + 5
        
        self.mixer.music.pause()
        
        self.save_start = start
        #print('Current Time in Song:', start)
        
        self.mixer.music.play(loops=0, start = start)
        
    def Backward_Song(self):
        start = self.save_start
        
        play_time = self.mixer.music.get_pos()

        start += (play_time/1000.0) - 5
        
        # mixer.music.pause()
        self.mixer.music.pause()
        
        self.save_start = start
        #print('Current Time in Song:', start)
        
        self.mixer.music.play(loops=0, start = start)
        
    def set_volume(self):
        volume = float(self.vol) / 100
        self.mixer.music.set_volume(volume)
        
 
class PageSix(tk.Frame): 

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Create five columns
        for i in range(7):
            self.grid_columnconfigure(i, weight=1, uniform="equal")
        
        # Place widgets using grid()
        label1 = tk.Label(self, text=" ")
        label1.grid(row=0, column=0)
        
        label2 = tk.Label(self, text=" ")
        label2.grid(row=0, column=1)
        
        label3 = tk.Label(self, text=" ")
        label3.grid(row=0, column=2)
        
        label4 = tk.Label(self, text=" ")
        label4.grid(row=0, column=3)
        
        label5 = tk.Label(self, text=" ")
        label5.grid(row=0, column=4)
        
        label6 = tk.Label(self, text=" ")
        label6.grid(row=0, column=5)
        
        label7 = tk.Label(self, text=" ")
        label7.grid(row=0, column=6)
        
        row=1
        
        label = tk.Label(self, text="Synthesizer", font=controller.title_font)
        label.grid(row=row, column=2, columnspan=3)
        row+=1
        
        #maximum number of sections (that can be parameterized)
        self.max_sections = 5
        
        ############################# Customize Parameters ###############################
        def toggle_buttons():
            if self.custom.get():
                synth_af.config(state=tk.NORMAL)
                synth_gr.config(state=tk.NORMAL)
                synth_ip.config(state=tk.NORMAL)
                
                menu_af.config(state=tk.NORMAL)
                menu_gr.config(state=tk.NORMAL)
                menu_ip.config(state=tk.NORMAL)
            else:
                synth_af.config(state=tk.DISABLED)
                synth_gr.config(state=tk.DISABLED)
                synth_ip.config(state=tk.DISABLED)
                
                menu_af.config(state=tk.DISABLED)
                menu_gr.config(state=tk.DISABLED)
                menu_ip.config(state=tk.DISABLED)
                
                
        # Create a boolean variable to store the checkbox value
        self.custom = tk.BooleanVar()
        self.custom.set(False)
        
        # Create the checkbox
        custom_checkbox = tk.Checkbutton(self, text="Customize Parameters",
                                         variable = self.custom,
                                         command=toggle_buttons)
        
        # Position the checkbox and filename entry widgets in the GUI window
        custom_checkbox.grid(row=row, column=1, columnspan=3, sticky="w", padx=10, pady=10)
        
        info0 = tk.Button(self, text=" Info* ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_CUSTOMIZE_PARAMETERS"))
        info0.grid(row=row, column=2)
        
        row+=1
        
        ############################# Synth Parameters ##################################
        param_frame = tk.Frame(self, padx=10, 
                             highlightbackground="black", highlightthickness=1)
        param_frame.grid(row=row, column=1, columnspan=5, pady=10, sticky='nsew')
        
        # Create five columns
        for i in range(5):
            param_frame.grid_columnconfigure(i, weight=1, uniform="equal")

        
        # Place widgets using grid()
        label1 = tk.Label(param_frame, text=" ")
        label1.grid(row=0, column=0)
        
        label2 = tk.Label(param_frame, text=" ")
        label2.grid(row=0, column=1)
        
        label3 = tk.Label(param_frame, text=" ")
        label3.grid(row=0, column=2)
        
        label4 = tk.Label(param_frame, text=" ")
        label4.grid(row=0, column=3)
        
        label5 = tk.Label(param_frame, text=" ")
        label5.grid(row=0, column=4)
        
        param_row=0
        label_synths = tk.Label(param_frame, text="Synth Units Parameters",
                                font=('Helvetica', 16))
        label_synths.grid(row=param_row, column=0, columnspan=3, sticky='w')
        param_row+=1
        
        self.af_param = list()
        self.gr_param = list()
        self.ip_param = list()
        
        synth_af = tk.Button(param_frame, text="Auto-Filter", state=tk.DISABLED,
                             command=lambda: controller.show_frame("PageSeven"))
        synth_gr = tk.Button(param_frame, text="Granular", state=tk.DISABLED,
                             command=lambda: controller.show_frame("PageEight"))
        synth_ip = tk.Button(param_frame, text="Interpolator", state=tk.DISABLED,
                             command=lambda: controller.show_frame("PageNine"))
        
        synth_pad=20
        synth_af.grid(row=param_row, column=0, pady=synth_pad, padx=20)
        synth_gr.grid(row=param_row, column=2, pady=synth_pad, padx=20)
        synth_ip.grid(row=param_row, column=4, pady=synth_pad, padx=20)
        row+=1
        
        ############################# Synth Instruments #################################
        inst_frame = tk.Frame(self, padx=10,
                             highlightbackground="black", highlightthickness=1)
        inst_frame.grid(row=row, column=1, columnspan=5, pady=10, sticky='nsew')
        
        # Create five columns
        for i in range(5):
            inst_frame.grid_columnconfigure(i, weight=1, uniform="equal")
        
        
        # Place widgets using grid()
        label12 = tk.Label(inst_frame, text=" ")
        label12.grid(row=0, column=0)
        
        label22 = tk.Label(inst_frame, text=" ")
        label22.grid(row=0, column=1)
        
        label32 = tk.Label(inst_frame, text=" ")
        label32.grid(row=0, column=2)
        
        label42 = tk.Label(inst_frame, text=" ")
        label42.grid(row=0, column=3)
        
        label52 = tk.Label(inst_frame, text=" ")
        label52.grid(row=0, column=4)
        
        inst_row=1
        label_inst = tk.Label(inst_frame, text="Choose Synths' Instruments",
                              font=('Helvetica', 16))
        label_inst.grid(row=param_row, column=0, columnspan=3, sticky='w')
        inst_row+=1
        
        
        OPTIONS = ["Bass", "Harmony", "Melody"]
        
        def check_combination(menu_vars, button):
            if len(set(menu_vars)) == len(menu_vars):
                button.config(state="normal")
            else:
                button.config(state="disabled")
        
        self.inst_af = tk.StringVar(value="Bass")
        self.inst_gr = tk.StringVar(value="Harmony")
        self.inst_ip = tk.StringVar(value="Melody")
        
        def set_inst_af(self, value):
            self.inst_af.set(str(value))
        def set_inst_gr(self, value):
            self.inst_gr.set(str(value))
        def set_inst_ip(self, value):
            self.inst_ip.set(str(value))
            
        menu_af = tk.OptionMenu(inst_frame, self.inst_af, *OPTIONS, 
                                command= combine_funcs(lambda selected: check_combination([self.inst_af.get(),
                                self.inst_gr.get(), self.inst_ip.get()], self.start_synth),
                                lambda value: set_inst_af(self,value)))
        
        
        menu_gr = tk.OptionMenu(inst_frame, self.inst_gr, *OPTIONS, 
                                command= combine_funcs(lambda selected: check_combination([self.inst_af.get(),
                                self.inst_gr.get(), self.inst_ip.get()], self.start_synth),
                                lambda value: set_inst_gr(self,value)))
        

        menu_ip = tk.OptionMenu(inst_frame, self.inst_ip, *OPTIONS, 
                                command= combine_funcs(lambda selected: check_combination([self.inst_af.get(),
                                self.inst_gr.get(), self.inst_ip.get()], self.start_synth),
                                lambda value: set_inst_ip(self,value)))
        
        menu_af.config(state="disabled")
        menu_gr.config(state="disabled")
        menu_ip.config(state="disabled")
        
        pad_menu=20
        menu_af.grid(row=inst_row, column=0, pady=pad_menu, padx=20)
        menu_gr.grid(row=inst_row, column=2, pady=pad_menu, padx=20)
        menu_ip.grid(row=inst_row, column=4, pady=pad_menu, padx=20)
        
        initial_values = [self.inst_af.get(), self.inst_gr.get(), self.inst_ip.get()]
        if len(set(initial_values)) == len(initial_values):
            initial_state = "normal"
        else:
            initial_state = "disabled"
        
        self.t_synth=threading.Thread(target=self.run_synth) 
        
        self.start_synth = tk.Button(self, text="Start Synthesis",
                                     font=('Helvetica', 18), state=initial_state,
                    command=combine_funcs(lambda: self.start_synth.config(text="Creating...",
                                                                          state= tk.DISABLED, width = 40),
                                                            self.Threading))#controller.show_frame("PageTen"))
        self.start_synth.config(width=20)

        row+=1
        
        self.start_synth.grid(row=row, column=1, columnspan=5, pady=20)
        row+=1

        #################################################################################################
        
    def Threading(self):
        # Call work function
        self.event = threading.Event()

        self.t_synth.start()
        
    def run_synth(self):
        
        #get synths vector (which synth was chosen for each instrument) #############################
        synth_id={'Bass': 0, 'Harmony': 1, 'Melody': 2}
        synths=[-1,-1,-1]
        
        #discover idx of that synth, in the order bass harmony melody
        idx_af = synth_id[str(self.inst_af.get())]
        idx_gr = synth_id[str(self.inst_gr.get())]
        idx_ip = synth_id[str(self.inst_ip.get())]
        
        #for that idx insert the code of that synth-> 0: af; 1:gr, 2:ip
        synths[idx_af] = 0
        synths[idx_gr] = 1
        synths[idx_ip] = 2
        
        #########################  ALOCATE PARAMETERS FOR IND_SYNTH  ################################
        if self.custom.get():
            af_param = self.af_param
            gr_param = self.gr_param
            ip_param = self.ip_param
            
            params_synths = [af_param, gr_param, ip_param]
            Ind_synth = [-1,-1,-1]

            idxs = [idx_af, idx_gr, idx_ip]
            #place synth params in the correct instrument: Ind_param[param_bass, param_harmony, param_melody]
            for idx, params in zip(idxs, params_synths):
                Ind_synth[idx] = params
        else: Ind_synth=list()
        #############################################################################################
        
        user_info = self.controller.frames['PageFour'].in_gen2['user_info']
        midi_info_out = self.controller.frames['PageFour'].in_gen2['midi_info_out']
        info_GA = self.controller.frames['PageFour'].in_gen1['info_GA']
        sample_info = self.controller.frames['PageFour'].in_gen2['sample_info']
        Ind_sample = self.controller.frames['PageFour'].out_ga['sample_Ind']
        midi_info_real = self.controller.frames['PageFour'].in_gen2['midi_info_real']
        
        from main import main_synth_GA
        synth_path = main_synth_GA(user_info, midi_info_out, info_GA, sample_info,
                                   Ind_sample, Ind_synth, synths, midi_info_real, self.start_synth)
        
        self.out_synth_path = synth_path
   
        
        self.start_synth.configure(text = "Listen to Arrangement!", state= tk.NORMAL, width = 25,
                               command = lambda: self.controller.show_frame("PageTen"))
            
class PageSeven(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Create five columns
        for i in range(5):
            if i!=2:
                self.grid_columnconfigure(i, weight=1, uniform="equal")
        
        # Place widgets using grid()
        label1 = tk.Label(self, text=" ")
        label1.grid(row=0, column=0)
        
        label2 = tk.Label(self, text=" ")
        label2.grid(row=0, column=1)
        
        label3 = tk.Label(self, text=" ")
        label3.grid(row=0, column=2)
        
        label4 = tk.Label(self, text=" ")
        label4.grid(row=0, column=3)
        
        label5 = tk.Label(self, text=" ")
        label5.grid(row=0, column=4)
        
        row = 1
        
        slider_length = 300
        row+=1
        
        row=0
        
        label = tk.Label(self, text="Auto-Filter", font=controller.title_font)
        #label.pack(side="top", fill="x", pady=10)
        label.grid(row=row, column=1 , columnspan=3, pady=20, sticky="nsew")
        row+=1
        
        #################################   INIT VARIABLES   ###################################
        #cutoff_floor=50
        self.cutoff_floor = tk.DoubleVar()
        self.cutoff_ceiling = tk.DoubleVar()
        self.lfo_floor = tk.DoubleVar()
        self.lfo_ceiling = tk.DoubleVar()
        self.lfo_shape = tk.StringVar(value="sine")
        self.lfo_evo = tk.StringVar(value="random")
        self.high_pass = tk.BooleanVar()
        
        self.cutoff_floor.set(20) 
        self.cutoff_ceiling.set(10_000) 
        self.lfo_floor.set(1) 
        self.lfo_ceiling.set(8) 

        self.high_pass.set(False)
        
        self.min_cutoff_floor = 8
        self.max_cutoff_floor = 199
        self.min_cutoff_ceiling = 200
        self.max_cutoff_ceiling = 20_000
        self.min_lfo_floor = 0.01
        self.max_lfo_floor = 4.9
        self.min_lfo_ceiling = 5
        self.max_lfo_ceiling = 10
        
        
        def set_cutoff_floor(self, value):
            self.cutoff_floor.set(float(value))
        def set_cutoff_ceiling(self, value):
            self.cutoff_ceiling.set(float(value))
        def set_lfo_floor(self, value):
            self.lfo_floor.set(float(value))
        def set_lfo_ceiling(self, value):
            self.lfo_ceiling.set(float(value))
        def set_lfo_shape(self, value):
            self.lfo_shape.set(str(value))
        def set_lfo_evo(self, value):
            self.lfo_evo.set(str(value))
        def set_high_pass(self, value):
            self.high_pass.set(bool(value))
        
        ##########################################   SLIDERS #######################################
        
        cutoff_floor_slider = tk.Scale(self, variable=self.cutoff_floor, from_= self.min_cutoff_floor,
                                       to=self.max_cutoff_floor, resolution=1, orient = 'horizontal',
                                       length = slider_length,
                                       command= lambda value: set_cutoff_floor(self,value))
        
        cutoff_ceiling_slider = tk.Scale(self, variable=self.cutoff_ceiling, from_= self.min_cutoff_ceiling,
                                         to=self.max_cutoff_ceiling, resolution=1, orient = 'horizontal',
                                         length = slider_length,
                                         command= lambda value: set_cutoff_ceiling(self,value))
        
        lfo_floor_slider = tk.Scale(self, variable=self.lfo_floor, from_= self.min_lfo_floor, 
                                    to= self.max_lfo_floor, resolution=0.1, orient = 'horizontal',
                                    length = slider_length,
                                    command= lambda value: set_lfo_floor(self,value))
        
        lfo_ceiling_slider = tk.Scale(self, variable=self.lfo_ceiling, from_= self.min_lfo_ceiling,
                                      to=self.max_lfo_ceiling, resolution=0.1, orient = 'horizontal',
                                      length = slider_length,
                                      command= lambda value: set_lfo_ceiling(self,value))
        
        #########################################   MENUS    ####################################
        shapes = ['sine', 'square', 'triangle', 'sawl', 'sawr']
        evos = ['constant', 'linear_up', 'linear_down', 'exp_up', 'exp_down', 'random']
        
        menu_lfo_shape = tk.OptionMenu(self, self.lfo_shape, *shapes, 
                                       command= lambda value: set_lfo_shape(self,value))
        menu_lfo_evo = tk.OptionMenu(self, self.lfo_evo, *evos, 
                                     command= lambda value: set_lfo_evo(self,value))
        
        high_pass_checkbox = tk.Checkbutton(self, text="High-Pass Filter", variable=self.high_pass, anchor='w',
                                             command= lambda value: set_high_pass(self, value)) 

        #########################################   GRID   ####################################
        ##   CUTOFF FLOOR    ##########################
        
        label_cutoff_floor = tk.Label(self, text='Cutoff Minimum', font=("Helvetica", "12"),
                                      justify="left")
        label_cutoff_floor.grid(row=row, column=0, columnspan=2, sticky = 'SW', padx=20)
        
        info1 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_CUTOFF"))
        info1.grid(row=row, column=1, sticky = 'S')
         
        cutoff_floor_slider.grid(row=row, column=2, columnspan=2)
        
        cutoff_floor_min = tk.Label(self, text = str(self.min_cutoff_floor),
                                    font=('Helvetica', "10"))
        cutoff_floor_min.grid(row=row, column=1, sticky = 'SE')
        
        cutoff_floor_max = tk.Label(self, text=str(self.max_cutoff_floor)+ " Hz",
                                    font=('Helvetica', "10"))
        cutoff_floor_max.grid(row=row, column=4, sticky = 'SW')
        row+=1
        
        ##   CUTOFF CEILING   ##########################
        
        label_cutoff_ceiling = tk.Label(self, text='Cutoff Maximum', font=("Helvetica", "12"),
                                        justify="left")
        label_cutoff_ceiling.grid(row=row, column=0, columnspan=2, sticky = 'SW', padx=20)
        
        info2 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_CUTOFF"))
        info2.grid(row=row, column=1, sticky = 'S')
        
        
        cutoff_ceiling_slider.grid(row=row, column=2, columnspan=2)
        
        cutoff_ceiling_min = tk.Label(self, text = str(self.min_cutoff_ceiling),
                                    font=('Helvetica', "10"))
        cutoff_ceiling_min.grid(row=row, column=1, sticky = 'SE')
        
        cutoff_ceiling_max = tk.Label(self, text=str(20)+'K'+ " Hz",
                                    font=('Helvetica', "10"))
        cutoff_ceiling_max.grid(row=row, column=4, sticky = 'SW')
        row+=1
        
        ##   LFO FLOOR   ##########################
        
        label_lfo_floor = tk.Label(self, text='LFO Minimum', font=("Helvetica", "12"),
                                        justify="left")
        label_lfo_floor.grid(row=row, column=0, columnspan=2, sticky = 'SW', padx=20)
        
        info3 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_LFO"))
        info3.grid(row=row, column=1, sticky = 'S')
        
        lfo_floor_slider.grid(row=row, column=2, columnspan=2)
        
        lfo_floor_min = tk.Label(self, text = str(self.min_lfo_floor),
                                    font=('Helvetica', "10"))
        lfo_floor_min.grid(row=row, column=1, sticky = 'SE')
        
        lfo_floor_max = tk.Label(self, text=str(self.max_lfo_floor)+ " Hz",
                                    font=('Helvetica', "10"))
        lfo_floor_max.grid(row=row, column=4, sticky = 'SW')
        row+=1
        
        ##   LFO CEILING   ##########################
        
        label_lfo_ceiling = tk.Label(self, text='LFO Maximum', font=("Helvetica", "12"),
                                        justify="left")
        label_lfo_ceiling.grid(row=row, column=0, columnspan=2, sticky = 'SW', padx=20)
        
        info4 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_LFO"))
        info4.grid(row=row, column=1, sticky = 'S')
        
        lfo_ceiling_slider.grid(row=row, column=2, columnspan=2)
        
        lfo_ceiling_min = tk.Label(self, text = str(self.min_lfo_ceiling),
                                    font=('Helvetica', "10"))
        lfo_ceiling_min.grid(row=row, column=1, sticky = 'SE')
        
        lfo_ceiling_max = tk.Label(self, text=str(self.max_lfo_ceiling) + " Hz",
                                    font=('Helvetica', "10"))
        lfo_ceiling_max.grid(row=row, column=4, sticky = 'SW')
        row+=1
        
        ########### lfo shape & lfo evo ##########
        
        label_lfo_shape = tk.Label(self, text='LFO Shape & EVO', font=("Helvetica", "11"),
                                        justify="left")
        label_lfo_shape.grid(row=row, column=0, columnspan=2, sticky = 'SW', padx=20, pady=10)
        
        info5 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_LFO_SHAPE_EVO"))
        info5.grid(row=row, column=1, sticky = 'S', pady=10)
        
        menu_lfo_shape.grid(row=row, column=2, sticky='S', pady=10)      
        menu_lfo_evo.grid(row=row, column=3, sticky='SW', pady=10)
        
        high_pass_checkbox.grid(row=row, column=3, sticky='SE', pady=10)
        row+=1
        
        go_back = tk.Button(self, text="Save Parameters", font=('Helvetica', 14),
                             command= combine_funcs(self.save_params, lambda: controller.show_frame("PageSix")))
        go_back.grid(row=row, column=4, pady=20, columnspan=2, sticky='S', padx=20)
        row+=1
        
        
    def save_params(self):
        #normalizar
        cutoff_floor = (self.cutoff_floor.get() - self.min_cutoff_floor) / (self.max_cutoff_floor - self.min_cutoff_floor)
        cutoff_ceiling = (self.cutoff_ceiling.get() - self.min_cutoff_ceiling) / (self.max_cutoff_ceiling - self.min_cutoff_ceiling)
        lfo_floor = (self.lfo_floor.get() - self.min_lfo_floor) / (self.max_lfo_floor - self.min_lfo_floor)
        lfo_ceiling = (self.lfo_ceiling.get() - self.min_lfo_ceiling) / (self.max_lfo_ceiling - self.min_lfo_ceiling)        
        
        
        af_param = [[float(cutoff_floor), float(cutoff_ceiling), float(lfo_floor),
                    float(lfo_ceiling), str(self.lfo_shape.get()), str(self.lfo_evo.get()), bool(self.high_pass.get())]]
        af_param *= self.controller.frames['PageSix'].max_sections
        
        
        self.controller.frames['PageSix'].af_param = af_param
        
        
class PageEight(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Create five columns
        for i in range(7):
            self.grid_columnconfigure(i, weight=1, uniform="equal")
        
        # Place widgets using grid()
        label1 = tk.Label(self, text=" ")
        label1.grid(row=0, column=0)
        
        label2 = tk.Label(self, text=" ")
        label2.grid(row=0, column=1)
        
        label3 = tk.Label(self, text=" ")
        label3.grid(row=0, column=2)
        
        label4 = tk.Label(self, text=" ")
        label4.grid(row=0, column=3)
        
        label5 = tk.Label(self, text=" ")
        label5.grid(row=0, column=4)
        
        label6 = tk.Label(self, text=" ")
        label6.grid(row=0, column=5)
        
        label7 = tk.Label(self, text=" ")
        label7.grid(row=0, column=6)
        
        row = 1
        
        slider_length = 350
        
        label = tk.Label(self, text="Granular", font=controller.title_font)
        label.grid(row=row, column=2 , columnspan=3, pady=20, sticky="nsew")
        row+=1
        
        ##############################   init variables    ##########################
        
        self.grain_size = tk.DoubleVar()
        self.grain_space = tk.DoubleVar()
        self.order = tk.BooleanVar()
        self.smoothing = tk.BooleanVar()
        self.sync = tk.BooleanVar()
        
        self.grain_size.set(0.005) 
        self.grain_space.set(0.005) 
        
        self.min_grain_size = 0.001
        self.max_grain_size = 0.01
        self.min_grain_space = 0.001
        self.max_grain_space = 0.01
        
        def set_grain_size(self, value):
            self.grain_size.set(float(value))
        def set_grain_space(self, value):
            self.grain_space.set(float(value))
        def set_order(self, value):
            self.order.set(bool(value))
        def set_smoothing(self, value):
            self.smoothing.set(bool(value))
        def set_sync(self, value):
            self.sync.set(bool(value))
            
        #####################################    grain size   ############################
        
        label_grain_size = tk.Label(self, text='Grain Size', font=("Helvetica", "12"),
                                        justify="left")
        label_grain_size.grid(row=row, column=0, columnspan=2, sticky = 'SW', padx=20)
        
        info1 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_GRAIN_SIZE"))
        info1.grid(row=row, column=1, sticky = 'SE')
        
        grain_size_slider = tk.Scale(self, variable=self.grain_size, from_= self.min_grain_size,
                                     to=self.max_grain_size, resolution=0.001, length= slider_length,
                                     orient = 'horizontal', command= lambda value: set_grain_size(self, value))
        
        grain_size_slider.grid(row=row, column=3, columnspan=3)
        
        grain_size_min = tk.Label(self, text = str(self.min_grain_size),
                                    font=('Helvetica', "10"))
        grain_size_min.grid(row=row, column=2, sticky = 'SE')
        
        grain_size_max = tk.Label(self, text=str(self.max_grain_size) + " s",
                                    font=('Helvetica', "10"))
        grain_size_max.grid(row=row, column=6, sticky = 'SW')
        
        row+=1
        
        ###################################    grain_space    ##############################
        
        label_grain_space = tk.Label(self, text='Grain Space', font=("Helvetica", "12"),
                                        justify="left")
        label_grain_space.grid(row=row, column=0, columnspan=2, sticky = 'SW', padx=20)
        
        info2 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_GRAIN_SPACE"))
        info2.grid(row=row, column=1, sticky = 'SE')
        
        grain_space_slider = tk.Scale(self, variable=self.grain_space, from_=self.min_grain_space,
                                      to=self.max_grain_space, resolution=0.001, length= slider_length,
                                      orient = 'horizontal', command= lambda value: set_grain_space(self, value))
        
        grain_space_slider.grid(row=row, column=3, columnspan=3)
        
        grain_space_min = tk.Label(self, text = str(self.min_grain_space),
                                    font=('Helvetica', "10"))
        grain_space_min.grid(row=row, column=2, sticky = 'SE')
        
        grain_space_max = tk.Label(self, text=str(self.max_grain_space) + " s",
                                    font=('Helvetica', "10"))
        grain_space_max.grid(row=row, column=6, sticky = 'SW')

        row+=1
        
        ##########################   checkboxes #########################################
        check_size=10
        
        ### order
        label_grain_order = tk.Label(self, text='Grain Order', font=("Helvetica", "12"),
                                        justify="left")
        label_grain_order.grid(row=row, column=0, columnspan=2, sticky = 'W', padx=20)
        
        info3 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_GRAIN_ORDER"))
        info3.grid(row=row, column=1, sticky = 'SE', pady=15)
        
        
        order_checkbox = tk.Checkbutton(self, variable=self.order,
                                        command= lambda value: set_order(self, value))
        
        order_checkbox.grid(row=row, column=1, sticky = 'S', pady = 15)
        
        
        ### smoothing
        
        label_smoothing = tk.Label(self, text='Smoothing', font=("Helvetica", "12"),
                                        justify="left")
        label_smoothing.grid(row=row, column=2, columnspan=2, sticky = 'W', padx=20)
        
        info4 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_SMOOTHING"))
        info4.grid(row=row, column=3, sticky = 'SE', pady=15)
        
        
        smoothing_checkbox = tk.Checkbutton(self, variable=self.smoothing,
                                         command= lambda value: set_smoothing(self, value))
        
        smoothing_checkbox.grid(row=row, column=3, sticky = 'S', pady = 15)
        
        
        ### even spacing
        
        label_spacing = tk.Label(self, text='Even Spacing', font=("Helvetica", "12"),
                                        justify="left")
        label_spacing.grid(row=row, column=4, columnspan=2, sticky = 'nsew', padx=10)
        
        info4 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_EVEN_SPACING"))
        info4.grid(row=row, column=5, sticky = 'SE', pady=15)
        
        
        sync_checkbox = tk.Checkbutton(self, variable=self.sync, 
                                       command= lambda value: set_sync(self, value))
        
        sync_checkbox.grid(row=row, column=5, sticky = 'S', pady = 15)
        
        row+=1
        
        go_back = tk.Button(self, text="Save Parameters", font=('Helvetica', 14),
                             command= combine_funcs(self.save_params, lambda: controller.show_frame("PageSix")))
        go_back.grid(row=row, column=5, pady=20, columnspan=2, sticky='S')
    
        
    def save_params(self):
        #normalize (0 to 1)
        grain_size = (self.grain_size.get() - self.min_grain_size) / (self.max_grain_size - self.min_grain_size)
        grain_space = (self.grain_space.get() - self.min_grain_space) / (self.max_grain_space - self.min_grain_space)
        
        gr_param = [[float(grain_size), float(grain_space), bool(self.order.get()),
                    bool(self.smoothing.get()), bool(self.sync.get())]]
        gr_param *= self.controller.frames['PageSix'].max_sections
        
        self.controller.frames['PageSix'].gr_param = gr_param

class PageNine(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
                
        # Create five columns
        for i in range(7):
            self.grid_columnconfigure(i, weight=1, uniform="equal")
        
        # Place widgets using grid()
        label1 = tk.Label(self, text=" ")
        label1.grid(row=0, column=0)
        
        label2 = tk.Label(self, text=" ")
        label2.grid(row=0, column=1)
        
        label3 = tk.Label(self, text=" ")
        label3.grid(row=0, column=2)
        
        label4 = tk.Label(self, text=" ")
        label4.grid(row=0, column=3)
        
        label5 = tk.Label(self, text=" ")
        label5.grid(row=0, column=4)
        
        label6 = tk.Label(self, text=" ")
        label6.grid(row=0, column=5)
        
        label7 = tk.Label(self, text=" ")
        label7.grid(row=0, column=6)
        
        row = 1
        
        label = tk.Label(self, text="Interpolator", font=controller.title_font)
        label.grid(row=row, column=2, columnspan=3, pady=20, sticky="nsew")
        row+=1
        
        
        self.signal_shape = tk.StringVar(value="sine")
        self.signal_evo = tk.StringVar(value="random")
        
        def set_signal_shape(self, value):
            self.signal_shape.set(str(value))
        def set_signal_evo(self, value):
            self.signal_evo.set(str(value))

        shapes = ['sine', 'square', 'triangle', 'sawl', 'sawr']
        evos = ['constant', 'linear_up', 'linear_down', 'exp_up', 'exp_down', 'random']
        
        menu_signal_shape = tk.OptionMenu(self, self.signal_shape, *shapes,
                                          command= lambda value: set_signal_shape(self, value))
        menu_signal_evo = tk.OptionMenu(self, self.signal_evo, *evos,
                                        command= lambda value: set_signal_evo(self, value))


        label_signal = tk.Label(self, text='Signal Shape & "EVO"', font=("Helvetica", "12"),
                                        justify="left")
        label_signal.grid(row=row, column=0, columnspan=2, sticky = 'SW', padx=20, pady=10)
        
        info1 = tk.Button(self, text=" * ", font = ("Helvetica", "10", "bold"),
                           command=lambda: controller.show_frame("INFO_SIGNAL_SHAPE_EVO"))
        info1.grid(row=row, column=2, sticky = 'SW', pady=10)
        
        menu_signal_shape.grid(row=row, column=3, sticky='S', pady=10)      
        menu_signal_evo.grid(row=row, column=4, sticky='SW', pady=10)        
        
        row+=1
        
        go_back = tk.Button(self, text="Save Parameters", font=('Helvetica', 14),
                             command= combine_funcs(self.save_params, lambda: controller.show_frame("PageSix")))
        go_back.grid(row=row, column=5, pady=20, columnspan=2, sticky='S')
        
        
        
    def save_params(self):
        ip_param = [[str(self.signal_shape.get()), str(self.signal_evo.get())]]
        ip_param *= self.controller.frames['PageSix'].max_sections
        
        self.controller.frames['PageSix'].ip_param = ip_param   
        
    
class PageTen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        # Create five columns
        for i in range(7):
            self.grid_columnconfigure(i, weight=1, uniform="equal")

        
        # Place widgets using grid()
        label1 = tk.Label(self, text=" ")
        label1.grid(row=0, column=0)
        
        label2 = tk.Label(self, text=" ")
        label2.grid(row=0, column=1)
        
        label3 = tk.Label(self, text=" ")
        label3.grid(row=0, column=2)
        
        label4 = tk.Label(self, text=" ")
        label4.grid(row=0, column=3)
        
        label5 = tk.Label(self, text=" ")
        label5.grid(row=0, column=4)
        
        label6 = tk.Label(self, text=" ")
        label6.grid(row=0, column=5)
        
        label7 = tk.Label(self, text=" ")
        label7.grid(row=0, column=6)
        
        row=1
        
        label = tk.Label(self, text="Final Arrangement\n(with Synth)", font=controller.title_font)
        label.grid(row=row, column=0, columnspan=7, pady=10)
        row+=1
        
        main_path = self.controller.shared_data["main_path"]
        
        #create playlist mixer
        self.mixer = pygame.mixer
        self.mixer.init()


        ###################################### icons ####################################
        #icon_path = main_path + '/GUI/icons/'
        icon_path = main_path + '/icons/'
        
        play_img = Image.open(icon_path + 'play.png')
        pause_img = Image.open( icon_path + 'pause.png')
        stop_img = Image.open( icon_path + 'stop.png')
        backward_img = Image.open( icon_path + 'backward.png')
        forward_img = Image.open( icon_path + 'forward.png')
        
        play_img = play_img.resize((20, 20))
        pause_img = pause_img.resize((20, 20))
        stop_img = stop_img.resize((20, 20))
        backward_img = backward_img.resize((20, 20))
        forward_img = forward_img.resize((20, 20))
        
        play_icon = ImageTk.PhotoImage(image = play_img, master = controller)
        pause_icon = ImageTk.PhotoImage(image = pause_img, master = controller)
        stop_icon = ImageTk.PhotoImage(image = stop_img, master = controller)
        backward_icon = ImageTk.PhotoImage(image = backward_img, master = controller)
        forward_icon = ImageTk.PhotoImage(image = forward_img, master = controller)
        
        ########################################## mp3 #######################################
        mp3_frame = tk.Frame(self, padx=10, pady=10,
                             highlightbackground="black", highlightthickness=1)
        mp3_frame.grid(row=row, column=1, columnspan=5, pady=10, sticky='nsew')
        
        for i in range(5):
            mp3_frame.grid_columnconfigure(i, weight=1, uniform="equal")
        
        pady_mp3 = 20
        
        self.paused = False
        self.save_start = 0
        
        play_btn = tk.Button(mp3_frame, image=play_icon, borderwidth=0, command= self.Play_Song)
        pause_btn = tk.Button(mp3_frame, image=pause_icon, borderwidth=0, command= self.Pause_Song)
        stop_btn = tk.Button(mp3_frame, image=stop_icon, borderwidth=0, command= self.Stop_Song)
        backward_btn = tk.Button(mp3_frame, image=backward_icon, borderwidth=0, command= self.Backward_Song)
        forward_btn = tk.Button(mp3_frame, image=forward_icon, borderwidth=0, command= self.Forward_Song)
        
        play_btn.image = play_icon
        pause_btn.image = pause_icon
        stop_btn.image = stop_icon
        backward_btn.image = backward_icon
        forward_btn.image = forward_icon
        
        backward_btn.grid(row = row, column = 0, pady=pady_mp3, sticky='nsew')
        stop_btn.grid(row = row, column = 1, pady=pady_mp3, sticky='nsew')
        play_btn.grid(row = row, column = 2, pady=pady_mp3, sticky='nsew')
        pause_btn.grid(row = row, column = 3, pady=pady_mp3, sticky='nsew')
        forward_btn.grid(row = row, column = 4, pady=pady_mp3, sticky='nsew')
        
        row+=1
        
        ######################################## volume #######################################
        
        def set_volume(val):
            volume = float(val) / 100
            self.mixer.music.set_volume(volume)
        
        label_vol = tk.Label(mp3_frame, text="Volume", font=('Helvetica', 16))
        label_vol.grid(row=row, column=0, columnspan=5, sticky='s')
        row+=1
        
        vol_slider = tk.Scale(mp3_frame, from_=0, to=100, orient='horizontal', command= set_volume,
                              length= 300)
        vol_slider.set(50)
        vol_slider.grid(row = row, column=1, columnspan=3)#, pady=10)
        row+=1
        
        ######################################################################################
        
        def close_window():
            pygame.mixer.quit()
            self.controller.destroy()
        
        close_button = tk.Button(self, text="Close", font=('Helvetica', 18),
                                 command=close_window)
        close_button.grid(row = row, column=3, columnspan=1, pady=50)
        row+=1
    
    def Play_Song(self):
        song_path = self.controller.frames['PageSix'].out_synth_path
        
        self.mixer.music.load(song_path)
        self.mixer.music.play(loops=0)
    
    def Stop_Song(self):
        self.mixer.music.stop()
        
        self.start=0
        
    
    def Pause_Song(self):
        
        if self.paused:
            #unpause 
            self.mixer.music.unpause()
            self.paused = False
        else:
            #pause
            self.mixer.music.pause()
            self.paused = True
        
        
    def Forward_Song(self):
        
        start = self.save_start
        
        play_time = self.mixer.music.get_pos()

        start += (play_time/1000.0) + 5
        
        self.mixer.music.pause()
        
        self.save_start = start
        #print('Current Time in Song:', start)
        
        self.mixer.music.play(loops=0, start = start)
        
    def Backward_Song(self):
        start = self.save_start
        
        play_time = self.mixer.music.get_pos()

        start += (play_time/1000.0) - 5
        
        self.mixer.music.pause()
        
        self.save_start = start
        #print('Current Time in Song:', start)
        
        self.mixer.music.play(loops=0, start = start)
        
    def set_volume(self):
        volume = float(self.vol) / 100
        self.mixer.music.set_volume(volume)
        

class INFO_MOD_LEVEL(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Mod Level:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['Mod level']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageTwo"))
        button.pack(expand=True, padx=20)
        
class INFO_MOD_FEEDBACK(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Mod Feedback:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['Mod Feedback']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageTwo"))
        button.pack(expand=True, padx=20)
        
class INFO_SECTION_SIZE(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Section Size:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['Section Size']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageTwo"))
        button.pack(expand=True, padx=20)
        
class INFO_CUSTOMIZE_PARAMETERS(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Customize Parameters:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['Customize Parameters']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 12), wraplength=500)
        info.pack(expand=True, padx=20)
        
        text = GUI_txt['Auto Filter']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 12), wraplength=500)
        info.pack(expand=True, padx=20)
        
        text = GUI_txt['Granular']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 12), wraplength=500)
        info.pack(expand=True, padx=20)
        
        text = GUI_txt['Interpolator']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 12), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageSix"))
        button.pack(expand=True, padx=20)
        
        
        
class INFO_CUTOFF(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Minimum and Maximum Cutoff:",
                         font=controller.title_font, wraplength=500)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['Cutoff']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageSeven"))
        button.pack(expand=True, padx=20)
        
class INFO_LFO(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Minimum and Maximum LFO Frequency:",
                         font=controller.title_font, wraplength=500)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['LFO']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageSeven"))
        button.pack(expand=True, padx=20)
        
class INFO_LFO_SHAPE_EVO(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="LFO Shape & EVO:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['LFO SHAPE EVO']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        text1 = GUI_txt['SHAPE LEGEND']
        info1 = tk.Label(self, text=text1,
                              font=('Helvetica', 12), wraplength=500)
        info1.pack(expand=True, padx=20)
        
        text2 = GUI_txt['EVO LEGEND']
        info2 = tk.Label(self, text=text2,
                              font=('Helvetica', 12), wraplength=500)
        info2.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageSeven"))
        button.pack(expand=True, padx=20)
        
class INFO_GRAIN_SIZE(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Grain Size:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['Grain Size']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageEight"))
        button.pack(expand=True, padx=20)
        
class INFO_GRAIN_SPACE(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Grain Space:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['Grain Space']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageEight"))
        button.pack(expand=True, padx=20)
        
class INFO_GRAIN_ORDER(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Grain Order:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['Grain Order']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageEight"))
        button.pack(expand=True, padx=20)
        
class INFO_SMOOTHING(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Smoothing:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['Smoothing']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageEight"))
        button.pack(expand=True, padx=20)
        
class INFO_EVEN_SPACING(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Even Spacing:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['Even Spacing']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageEight"))
        button.pack(expand=True, padx=20)
        
class INFO_SIGNAL_SHAPE_EVO(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = tk.Label(self, text="Wavetable Shaepe & EVO:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        text = GUI_txt['SIGNAL SHAPE EVO']
        info = tk.Label(self, text=text,
                              font=('Helvetica', 14), wraplength=500)
        info.pack(expand=True, padx=20)
        
        
        button = tk.Button(self, text="Go Back", font=('Helvetica', 18),
                           command=lambda: controller.show_frame("PageNine"))
        button.pack(expand=True, padx=20)
        

        
def get_GUI_TEXT():

    GUI_txt = dict()
    
    GUI_txt['Mod level'] = 'Maximum Probability of MIDI Modification Ocurrence.\
        The bigger the value, the more likely to occur Modifications.'
        
    GUI_txt['Mod Feedback'] = 'Number of times the MIDI pattern gets processed by the MIDI Modification Unit.\
        A higher number of times will overlay MIDI Mod effects on top of each other.'
        
    GUI_txt['Section Size'] = 'Number of bars of a single "section" of the arrangement.\
        This determines the number of samples used: for each instrument there is one sample per section.'
        
    GUI_txt['Customize Parameters'] = 'There are three Synth modules: Auto-Filter, Granular and Interpolator.\
        Each Unit is assigned to one Instrument. If you choose the "Customize Parameters" option, you can pick which instruments has each Synth Unit, and customize the Synths Parameters. \
                Otherwise, dubgen will pick the parameters for you and assign each Synth Unit.'
                
    GUI_txt['Auto Filter'] ='Auto-Filter: Applies a Low/High Pass Filter, with a cutoff frequency modulated by an LFO.'
    GUI_txt['Granular'] ='Granular: Divides the signal into small segments (grains), creating new sounds by reordering, spacing out, or "condensing" those grains.'
    GUI_txt['Interpolator'] ='Interpolator: Interpolates the input signal with another signal stored in our wavetable.'
                
                
    GUI_txt['Cutoff'] = 'Determines the Maximum and Minimum Cutoff Frequency of the Auto Filter.\
        The Cutoff Frequency will then oscillate between those two values, modulated by the LFO.'
    
    GUI_txt['LFO'] = 'Determines the Maximum and Minimum LFO Frequency of the Auto Filter. \
        This will determine the rate at which the Cutoff Frequency oscillates between its Minimum and Maximum Value.'
            
    GUI_txt['LFO SHAPE EVO'] = 'Shape: Shape of the LFO signal. \
        EVO: Function that determines the evolution of the LFO frequency.'
        
    GUI_txt['SHAPE LEGEND'] = 'SHAPE legend:\n sawl/sawr: sawtooth wave with a left (l) or right (r) "teeth".'
        
    GUI_txt['EVO LEGEND'] = 'EVO Legend:\n linear_up/down: crescent or decrescent linear function;\n \
        exp_up/down: Exponetinal crescent or decrescent function.\n'
        
    GUI_txt['Grain Size'] = 'Size of each grain.'
    GUI_txt['Grain Space'] ='Space between each grain.'
    GUI_txt['Grain Order'] ='True means the grains maintain original order.\
        False means the grains are reordered.'
    GUI_txt['Smoothing'] = 'Smoothing Filter smoothens transition between grains.'
    GUI_txt['Even Spacing'] ='True means spaces between grains are all the same.\
        False means the space between grains varies randomly.'
    
    GUI_txt['SIGNAL SHAPE EVO'] = 'Determines the shape of the wavetable sound that will be interpolated with the input signal. EVO determines the evolution of the balance (interpolation weight from 0 to 1) between the two sounds.'
            
    return GUI_txt
        
if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    
    GUI_txt = get_GUI_TEXT()
    
    pygame.mixer.quit()
    
    app = App()
    
    app.lift()
    app.attributes('-topmost', True)
    app.after_idle(app.attributes,'-topmost', False)
    
    app.mainloop()
