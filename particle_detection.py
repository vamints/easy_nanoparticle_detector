import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os.path
import math
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import shutil
import tkinter as tk
from PIL import Image
from PIL.TiffTags import TAGS
import PIL.ExifTags

unit_dict = {'nm':1e-9,'um':1e-6,'mm':1e-3,'pm':1e-12}

def load_image():
    global h1,l2,canvas,pixel_size_label

    try:
        img = cv.imread(file_path.get())
        canvas.get_tk_widget().pack_forget() 
        fig, ax = plt.subplots(figsize=(10,10),dpi=300)
        ax.imshow(img)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(padx = 20,pady = (30,0))

        file_name = file_path.get()
        file_name = file_name.split('/')[:1][0]
        h1.config(text = file_name)
        l2.config(text = "")

        folder = file_path.get().replace('\\','/').split('/')
        if(folder[0] == "Results"):
            file_name = folder[1]
            h1.config(text = file_name+'.tif')
        
        else:        
            if not os.path.exists('Results'):
                os.makedirs('Results')


            results_folder = 'Results/'+file_name[:-4]

            if not os.path.exists(results_folder):
                os.makedirs(results_folder) 
            original_location = results_folder+"/original_"+file_name[:-4]+".tif"
            
            if(os.path.isfile(original_location) == True):
                restore_original()
            else:
                shutil.copyfile(file_path.get(), original_location) 
                
        file_name = h1.cget("text")        
        results_folder = 'Results/'+file_name[:-4]        
        original_location = results_folder+"/original_"+file_name[:-4]+".tif"                
        sourceimg = Image.open(original_location)
        exifdata = sourceimg.getexif()
        size = '0 nm'
        for k, v in exifdata.items():
            temp = str(v).split('\r\n')
            for part in temp:
                 if('Pixel Size' in part):
                    part = part.split(' = ')
                    if(part[0] == 'Pixel Size'):
                        size = part[1]
        pixel_size_label.config(text = "pixel width: "+size)
        
        
    except:
        l2.config(text = "file not found")   
    
    return

def soft_load_image():
    global h1,l2,canvas
    try:
        img = cv.imread(file_path.get())
        canvas.get_tk_widget().pack_forget() 
        fig, ax = plt.subplots(figsize=(10,10),dpi=300)
        ax.imshow(img)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(padx = 20,pady = (30,0))
        l2.config(text = "")
    except:
        l2.config(text = "file not found")
    return
    
def restore_original():
    file_name = h1.cget("text")
    results_folder = 'Results/'+file_name[:-4]
    w1.delete(0,tk.END)
    w1.insert(0,results_folder+"/original_"+file_name[:-4]+".tif")
    soft_load_image()
    return

def invert_image():
    global w1,canvas
    try:
        file_name = h1.cget("text")
        results_folder = 'Results/'+file_name[:-4]
        
        img = cv.imread(file_path.get())
        file_name = file_name.split('/')[-1]
        
        counter = 0
        new_file_path = results_folder+"/inverse_"+file_name[:-4]+"_"+str(counter)+".png"
        while(os.path.isfile(new_file_path) == True):
            counter += 1
            new_file_path = results_folder+"/inverse_"+file_name[:-4]+"_"+str(counter)+".png" 
        
        cv.imwrite( os.path.join(new_file_path), ~img );
        w1.delete(0,tk.END)
        w1.insert(0,new_file_path)
        soft_load_image()
    except:
        l2.config(text = "file not found")
    return

def crop_image():
    file_name = file_path.get()
    
    img = cv.imread(file_name)    
    y = img.shape[0]-v1.get()
    x = img.shape[1]
    crop = img[0:y, 0:x]
    file_name = file_name.split('/')[-1]
    
    original_file_name = h1.cget("text")
    results_folder = 'Results/'+original_file_name[:-4]
    
    counter = 0        
    new_file_path = results_folder+"/"+file_name[:-4]+"_croped_"+str(counter)+".png"
    while(os.path.isfile(new_file_path) == True):
        counter += 1
        new_file_path = results_folder+"/"+file_name[:-4]+"_croped_"+str(counter)+".png"
    w1.delete(0,tk.END)
    w1.insert(0,new_file_path)          
    cv.imwrite(os.path.join(new_file_path), crop)
    soft_load_image()
    return
    


def detect_particles():
    global canvas,canvas2    
    result_label.config(text = "Calculating Particles")
    
    file_name = file_path.get()    
    img = cv.imread(file_name)  
    
    params = cv.SimpleBlobDetector_Params()

    params.minThreshold = minThreshold.get()
    params.maxThreshold = maxThreshold.get()
    params.thresholdStep = thresholdStep.get()
    params.filterByConvexity = filterByConvexity.get()
    params.minConvexity = minConvexity.get()

    params.filterByInertia = filterByInertia.get()
    params.minInertiaRatio = minInertiaRatio.get()

    params.filterByArea = filterByArea.get()
    params.minArea = minArea.get()

    params.filterByCircularity = filterByCircularity.get()
    params.minCircularity = minCircularity.get()

    detector = cv.SimpleBlobDetector_create(params)
    particles = detector.detect(img)
    im_with_particles = cv.drawKeypoints(img, particles, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    
    file_name = file_name.split('/')[-1]
    
    original_file_name = h1.cget("text")
    results_folder = 'Results/'+original_file_name[:-4]   
    
    counter = 0    
    new_file_path = results_folder+"/detected_particles_"+file_name[:-4]+"_"+str(counter)+".png"
    while(os.path.isfile(new_file_path) == True):
        counter += 1
        new_file_path = results_folder+"/detected_particles_"+file_name[:-4]+"_"+str(counter)+".png"
        
    cv.imwrite( os.path.join(new_file_path), im_with_particles);
    
    metadata = {}
    metadata['source_file'] = file_path.get() 
    metadata['output_file'] = new_file_path
    metadata['minThreshold'] = minThreshold.get()
    metadata['maxThreshold'] = maxThreshold.get()
    metadata['thresholdStep'] = thresholdStep.get()
    metadata['filterByConvexity'] = filterByConvexity.get()
    metadata['minConvexity'] = minConvexity.get()
    metadata['filterByInertia'] = filterByInertia.get()
    metadata['minInertiaRatio'] = minInertiaRatio.get()
    metadata['filterByArea'] = filterByArea.get()
    metadata['minArea'] = minArea.get()
    metadata['filterByCircularity'] = filterByCircularity.get()
    metadata['minCircularity'] = minCircularity.get()
    pixelsize = str(pixel_size_label.cget("text")).split(': ')[1]

    pixelsizeunits = pixelsize.split(' ')[1]
    pixelsize = pixelsize.split(' ')[0]
    pixelsize = float(pixelsize)*unit_dict[pixelsizeunits]/unit_dict['nm']
    
    metadata['pixelsize'] = str(pixelsize)+' nm'
    print(pixelsize)
    metadata = pd.DataFrame.from_dict(metadata, orient='index')
    
    metadata_path = results_folder+"/metadata_"+file_name[:-4]+"_"+str(counter)+".csv"
    metadata.to_csv(metadata_path,header=False)
    
    particle_data = pd.DataFrame(columns=['coordinates','diameter (px)','size (px^2)','diameter (nm)','size (nm^2)'])
    particle_data['size'] = np.zeros(len(particles))
    for i in range(len(particle_data)):
        particle_data['coordinates'][i] = particles[i].pt
        particle_data['diameter (px)'][i] = particles[i].size
        particle_data['size (px^2)'][i] = math.pi*(particles[i].size/2)**2
        true_particle_size = pixelsize*particles[i].size
        particle_data['diameter (nm)'][i] = true_particle_size
        particle_data['size (nm^2)'][i] = math.pi*(true_particle_size/2)**2
        
    
    particle_data_path = results_folder+"/particle_data_"+file_name[:-4]+"_"+str(counter)+".csv"
    particle_data.to_csv(particle_data_path)
    
    canvas2.get_tk_widget().pack_forget() 
    fig2,ax2 = plt.subplots(figsize=(3,2))
    if pixelsize > 0:
        ax2.hist((particle_data['diameter (nm)'].to_numpy()),bins=25)
        ax2.set_xlabel('particle diameter (nm)')
    else:
        ax2.hist((particle_data['diameter (px)'].to_numpy()),bins=25)
        ax2.set_xlabel('particle diameter (px)')
    ax2.set_ylabel('frequency')
    fig2.tight_layout()
    canvas2 = FigureCanvasTkAgg(fig2, control_frame)
    canvas2.get_tk_widget().pack(pady = (10,0))
    
    canvas.get_tk_widget().pack_forget() 
    fig, ax = plt.subplots(figsize=(10,10),dpi=300)
    ax.imshow(im_with_particles)
    ax.axis('off')
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, root)
    canvas.get_tk_widget().pack(padx = 20,pady = (30,0))
    result_label.config(text = "Results saved in: "+new_file_path)
    return
    
root = tk.Tk()  
root.geometry("1200x1200") 
#variables in tkinter widget
file_path = tk.StringVar()
v1 = tk.IntVar(value=79)
pixel_size = tk.DoubleVar(value = 1)
#create image field
fig, ax = plt.subplots(figsize=(10,10))
fig = fig.clear()
canvas = FigureCanvasTkAgg(fig, root)


h1 = tk.Label(root, text='',font=("Arial", 32))
h1.pack(anchor=tk.N,padx = 50,pady = (10,0))

#data loader
control_frame = tk.Frame(root,width=160)

l1 = tk.Label(control_frame, text='File loader',font=("Arial", 16))
l1_1 = tk.Label(control_frame, text='File path')
w1 = tk.Entry(control_frame, textvariable=file_path,width=40,justify=tk.LEFT)
b1 = tk.Button(control_frame, text="load image", command=load_image)
l2 = tk.Label(control_frame, text="",fg='red')



l1.pack(anchor=tk.NW)
l1.pack(anchor=tk.NW)
w1.pack(anchor=tk.NW)
b1.pack(anchor=tk.NW)
l2.pack(anchor=tk.NW)


#preprocessing

l3 = tk.Label(control_frame, text="Preprocessing Steps",font=("Arial", 16))
b2 = tk.Button(control_frame, text="Original", command=restore_original,width=26)
b3 = tk.Button(control_frame, text="Invert Image", command=invert_image,width=26)


crop_frame = tk.Frame(control_frame,width=120)

s1 = tk.Scale(crop_frame, variable = v1, 
           from_ = 0, to = 150, resolution=1,
           orient = tk.HORIZONTAL)  
b4 = tk.Button(crop_frame, text="Crop Bottom", command=crop_image)


pixel_size_label = tk.Label(control_frame, text="pixel width:",justify=tk.LEFT)

l3.pack(anchor=tk.NW,pady = (10,0))
b2.pack(anchor=tk.NW)
b3.pack(anchor=tk.NW)

s1.pack(side=tk.LEFT,pady = (0,0))
b4.pack(side=tk.RIGHT,pady = (8,0))
crop_frame.pack(anchor=tk.NW)

pixel_size_label.pack(anchor=tk.NW)

#particle detector
minThreshold = tk.IntVar(value = 0) 
maxThreshold = tk.IntVar(value = 150) 
thresholdStep = tk.DoubleVar(value=0.5)

filterByConvexity = tk.BooleanVar(value=True)
minConvexity = tk.DoubleVar(value=0.87)

filterByInertia = tk.BooleanVar(value=True)
minInertiaRatio = tk.DoubleVar(value=0.08)

filterByArea = tk.BooleanVar(value=True)
minArea = tk.IntVar(value = 10) 

filterByCircularity = tk.BooleanVar(value=False)
minCircularity = tk.DoubleVar(value = 0.1) 

    
l4 = tk.Label(control_frame, text="Particle Detection",font=("Arial", 16))

threshold_frame_1 = tk.Frame(control_frame,width=120)
threshold_frame_2 = tk.Frame(control_frame,width=120)
threshold_frame_3 = tk.Frame(control_frame,width=120)
l5 = tk.Label(threshold_frame_1, text="Minimal Pixel \nIntensity Threshold",justify=tk.LEFT)
l5_1 = tk.Label(threshold_frame_2, text="Maximal Pixel \nIntensity Threshold",justify=tk.LEFT)
s2 = tk.Scale(threshold_frame_1, variable = minThreshold, from_ = 0, to = 255, resolution=1, orient = tk.HORIZONTAL)  
s2_1 = tk.Scale(threshold_frame_2, variable = maxThreshold, from_ = 0, to = 255, resolution=1, orient = tk.HORIZONTAL)  
e1 = tk.Entry(threshold_frame_1, textvariable=minThreshold,width=5)
e2 = tk.Entry(threshold_frame_2, textvariable=maxThreshold,width=5)

l5_2 = tk.Label(threshold_frame_3, text="Threshold \nStep",justify=tk.LEFT)
s2_2 = tk.Scale(threshold_frame_3, variable = thresholdStep, from_ = 0, to = 5, resolution=0.5, orient = tk.HORIZONTAL)  
e2_2 = tk.Entry(threshold_frame_3, textvariable=thresholdStep,width=5)

convexity_frame_1 = tk.Frame(control_frame,width=120)
convexity_frame_2 = tk.Frame(control_frame,width=120)

l6 = tk.Label(convexity_frame_1, text="Filter by\nConvexity",justify=tk.LEFT)
r1 = tk.Checkbutton(convexity_frame_1, variable=filterByConvexity)

l7 = tk.Label(convexity_frame_2, text="Minimal \nConvexity",justify=tk.LEFT)
s3 = tk.Scale(convexity_frame_2, variable = minConvexity, from_ = 0, to = 1, resolution=0.01, orient = tk.HORIZONTAL)  
e3 = tk.Entry(convexity_frame_2, textvariable=minConvexity,width=5)

inertia_frame_1 = tk.Frame(control_frame,width=120)
inertia_frame_2 = tk.Frame(control_frame,width=120)
l8 = tk.Label(inertia_frame_1, text="Filter by\nIntertia",justify=tk.LEFT)
l9 = tk.Label(inertia_frame_2, text="Minimal Inertia\nRatio",justify=tk.LEFT)
r8 = tk.Checkbutton(inertia_frame_1, variable=filterByConvexity)
s9 = tk.Scale(inertia_frame_2, variable = minInertiaRatio, from_ = 0, to = 1, resolution=0.01, orient = tk.HORIZONTAL)  
e9 = tk.Entry(inertia_frame_2, textvariable=minInertiaRatio,width=5)

area_frame_1 = tk.Frame(control_frame,width=120)
area_frame_2 = tk.Frame(control_frame,width=120)
l10 = tk.Label(area_frame_1, text="Filter by\nArea",justify=tk.LEFT)
l11 = tk.Label(area_frame_2, text="Minimal Area\nSize",justify=tk.LEFT)
r10 = tk.Checkbutton(area_frame_1, variable=filterByArea)
s11 = tk.Scale(area_frame_2, variable = minArea, from_ = 0, to = 100, resolution=1, orient = tk.HORIZONTAL)  
e11 = tk.Entry(area_frame_2, textvariable=minArea,width=5)

circularity_frame_1 = tk.Frame(control_frame,width=120)
circularity_frame_2 = tk.Frame(control_frame,width=120)
l12 = tk.Label(circularity_frame_1, text="Filter by\nCircularity",justify=tk.LEFT)
l13= tk.Label(circularity_frame_2, text="Minimal\nCircularity",justify=tk.LEFT)
r12 = tk.Checkbutton(circularity_frame_1, variable=filterByCircularity)
s13 = tk.Scale(circularity_frame_2, variable = minCircularity, from_ = 0, to = 1, resolution=0.01, orient = tk.HORIZONTAL)  
e13 = tk.Entry(circularity_frame_2, textvariable=minCircularity,width=5)

       

b5 = tk.Button(control_frame, text="Detect Particles", command=detect_particles)
result_label = tk.Label(control_frame, text="")


l4.pack(anchor=tk.NW,pady = (10,0))

l5.pack(side=tk.LEFT)
e1.pack(side=tk.RIGHT,pady=(17,0))
s2.pack(side=tk.RIGHT)
threshold_frame_1.pack(anchor=tk.NW)

l5_1.pack(side=tk.LEFT)
e2.pack(side=tk.RIGHT,pady=(17,0))
s2_1.pack(side=tk.RIGHT)
threshold_frame_2.pack(anchor=tk.NW)

l5_2.pack(side=tk.LEFT)
e2_2.pack(side=tk.RIGHT,pady=(17,0))
s2_2.pack(side=tk.RIGHT,padx=(45,0))
threshold_frame_3.pack(anchor=tk.NW)

l6.pack(side=tk.LEFT)
r1.pack(side=tk.RIGHT,padx=(45,0))
convexity_frame_1.pack(anchor=tk.NW)

l7.pack(side=tk.LEFT)
e3.pack(side=tk.RIGHT,pady=(17,0))
s3.pack(side=tk.RIGHT,padx=(47,0))
convexity_frame_2.pack(anchor=tk.NW)


l8.pack(side=tk.LEFT)
r8.pack(side=tk.RIGHT,padx=(56,0))
inertia_frame_1.pack(anchor=tk.NW)

l9.pack(side=tk.LEFT)
e9.pack(side=tk.RIGHT,pady=(17,0))
s9.pack(side=tk.RIGHT,padx=(20,0))
inertia_frame_2.pack(anchor=tk.NW)

l10.pack(side=tk.LEFT)
r10.pack(side=tk.RIGHT,padx=(56,0))
area_frame_1.pack(anchor=tk.NW)

l11.pack(side=tk.LEFT)
e11.pack(side=tk.RIGHT,pady=(17,0))
s11.pack(side=tk.RIGHT,padx=(29,0))
area_frame_2.pack(anchor=tk.NW)


l12.pack(side=tk.LEFT)
r12.pack(side=tk.RIGHT,padx=(44,0))
circularity_frame_1.pack(anchor=tk.NW)

l13.pack(side=tk.LEFT)
e13.pack(side=tk.RIGHT,pady=(17,0))
s13.pack(side=tk.RIGHT,padx=(46,0))
circularity_frame_2.pack(anchor=tk.NW)

b5.pack(anchor=tk.NW,pady=(10,0))
result_label.pack(anchor=tk.NW)

fig2, ax2 = plt.subplots(figsize=(0.2,0.2),dpi=72)
fig2 = fig2.clear()
canvas2 = FigureCanvasTkAgg(fig2, control_frame)

control_frame.pack(side=tk.LEFT,anchor=tk.NW,padx = 20,pady = (0,0),expand=False)
canvas.get_tk_widget().pack(side=tk.RIGHT,padx = 20,pady = (30,0)) 


root.mainloop()
