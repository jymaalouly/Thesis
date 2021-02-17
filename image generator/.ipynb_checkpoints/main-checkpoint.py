import tkinter as tk
import model
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
from image_proccessing import Generate
fields = 'Correlation', 'Density', 'Pos_u', 'Pos_v' , 'Variance','Scatter', 'Skew_y', 'Skew_x'
def makeslider(root, fields):
    entries = []
    for index,field in enumerate(fields):
        w = tk.Scale(root, from_=-10, to=10,  resolution=0.1, label = field , orient=tk.HORIZONTAL, command=encode_button)
        w.config(state=DISABLED,takefocus=0)
        w.set(0)
        w.grid(column=0, row= index+3 ,columnspan=1,sticky='NSEW')
        entries.append((field, w))
    return entries

def browse_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    file = tk.filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")) )
    folder_path.set(file)
    
    img2 = ImageTk.PhotoImage(resize(file))
    panel.configure(image=img2)
    panel.image = img2
    image.pre_processing(file)
    encode_button(0)
    button3.config(state='normal')
    num_images.config(state='normal')
    num_points.config(state='normal')
    for entry in slid:
        entry[1].config(state='normal')
        entry[1].set(0)
        
def encode_button(w):
    image.create_reconstrution(slid)
    result = None
    
    #os.remove("C:/Users/Tasiko/Thesis/image generator/temp/temp_reconstruction/temp_1.png")
            
    while result is None:
        try:
            img3 = ImageTk.PhotoImage(resize('C:/Users/Tasiko/Thesis/image generator/temp/temp_reconstruction/temp_1.png'))
            panel_2.configure(image=img3)
            panel_2.image = img3
            result = 1
        except:
             pass
def generate_button():
    Generate('C:/Users/Tasiko/Thesis/image generator/temp/temp_reconstruction/temp_1.png',num_images.get(),num_points.get())
    
def resize(path):
    image = Image.open(path)
    image = image.resize((200, 200), Image.ANTIALIAS)
    return image
root = tk.Tk()     


#w, h = root.winfo_screenwidth(), root.winfo_screenheight()
#root.geometry("%dx%d+0+0" % (w, h))


folder_path = StringVar()
image = model.Traversal() 
slid = makeslider(root, fields)
try:
    img = ImageTk.PhotoImage(resize('C:/Users/Tasiko/Thesis/image generator/image_not_available.png'))
    panel = tk.Label(root, image=img)
    panel.grid(column=0,row=1,padx = 10 ,pady=10,sticky='nesw')
    
    panel_2 = tk.Label(root, image=img)
    panel_2.grid(column=1,padx = 10, row=1, sticky='nesw')

except:
    0

button2 = Button(root,text="Browse", command=browse_button )
button2.grid(column=0,row=2, padx = 20 ,pady = 5 ,sticky='nesw' )

button3 = Button(root,text="Generate" , command=generate_button )
button3.config(state=DISABLED,takefocus=0)
button3.grid(column=1,row=2,padx = 20 ,pady = 5 ,  sticky='nesw')

num_images = tk.Scale(root, from_=1, to=200, label = 'Number of images' , orient=tk.HORIZONTAL)
num_images.config(state=DISABLED,takefocus=0)
num_images.grid(column=1, row= 3 ,columnspan=1,sticky='NSEW')
num_points = tk.Scale(root, from_=25, to=200 , label = 'Number of Data points' , orient=tk.HORIZONTAL)
num_points.config(state=DISABLED,takefocus=0)
num_points.grid(column=1, row= 4 ,columnspan=1,sticky='NSEW')


tk.mainloop()  








