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


def empty_messages():
    global canvas,canvas2
    coverage_results.config(text = "\n")
    l2.config(text = "")
    result_label.config(text = "")
    canvas.get_tk_widget().pack_forget() 
    canvas2.get_tk_widget().pack_forget() 
    return

def load_image():
    global h1,l2,canvas,pixel_size_label,canvas2
    canvas2.get_tk_widget().pack_forget() 
    empty_messages()
    try:
        img = cv.imread(file_path.get())
        canvas.get_tk_widget().pack_forget() 
        fig, ax = plt.subplots(figsize=(10,10),dpi=300)
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(padx = 20,pady = (30,0))

        file_name = file_path.get()
        file_name = file_name.split('/')[:1][0]
        h1.config(text = file_name)

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
            original_location = results_folder+"/0000_original_"+file_name[:-4]+".tif"

            if(os.path.isfile(original_location) == True):
                restore_original()
            else:
                shutil.copyfile(file_path.get(), original_location)
                restore_original()

        file_name = h1.cget("text")        
        results_folder = 'Results/'+file_name[:-4]        
        original_location = results_folder+"/0000_original_"+file_name[:-4]+".tif"   

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
        l2.config(text = "No file found")   
    
    return

def soft_load_image():
    global h1,l2,canvas
    empty_messages()
    try:
        img = cv.imread(file_path.get())
        canvas.get_tk_widget().pack_forget() 
        fig, ax = plt.subplots(figsize=(10,10),dpi=300)
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(padx = 20,pady = (30,0))
    except:
        l2.config(text = "file not found")
    return
    
def restore_original():
    empty_messages()
    try:        
        file_name = h1.cget("text")
        results_folder = 'Results/'+file_name[:-4]
        w1.delete(0,tk.END)
        w1.insert(0,results_folder+"/0000_original_"+file_name[:-4]+".tif")
        soft_load_image()        
    except:
        l2.config(text = "No file found")
    return

def get_unique_prefix(loc):
    counter = 0
    unique = False    
    arr = os.listdir(loc)
    
    for file in arr:
        try:
           counter = max(int(file[0:4])+1,counter)
        except:
            print('no_prefix found')

    while(not unique):
        unique=True
        prefix = '{0}'.format(str(counter).zfill(4))+'_'
        for file in arr:
            if(prefix == file[0:5]):
                counter += 1
                unique=False
            
    return '{0}'.format(str(counter).zfill(4))


def invert_image():
    global w1,canvas
    empty_messages()
    try:        
        source_file_name = h1.cget("text")[:-4]
        results_folder = 'Results/'+source_file_name
        
        file_name = file_path.get()
        img = cv.imread(file_name)
        file_name = file_name.split('/')[-1][:-4]
        if('_croped' == file_name[-7:]):
            source_file_name += '_croped'
        
        prefix = get_unique_prefix(results_folder)
        
        new_file_path = results_folder+"/"+prefix+"_inverse_"+source_file_name+".png"

        
        cv.imwrite( os.path.join(new_file_path), ~img );
        w1.delete(0,tk.END)
        w1.insert(0,new_file_path)
        result_label.config(text = "Results saved in: "+new_file_path)
        soft_load_image()
    except:
        l2.config(text = "No file found")
    return

def crop_image():
    empty_messages()
    try:        
        file_name = file_path.get()

        img = cv.imread(file_name)    
        y = img.shape[0]-v1.get()
        x = img.shape[1]
        crop = img[0:y, 0:x]
        file_name = file_name.split('/')[-1]

        source_file_name = h1.cget("text")[:-4]
        results_folder = 'Results/'+source_file_name

        file_name = file_name.split('/')[-1]
        if('_inverse_' == file_name[4:13]):
            source_file_name = 'inverse_'+source_file_name

        prefix = get_unique_prefix(results_folder)

        new_file_path = results_folder+"/"+prefix+'_'+source_file_name+"_croped.png"


        w1.delete(0,tk.END)
        w1.insert(0,new_file_path)          
        cv.imwrite(os.path.join(new_file_path), crop)
        result_label.config(text = "Results saved in: "+new_file_path)
        soft_load_image()
    except:
        l2.config(text = "No file found")
    return


def detect_particles():
    global canvas,canvas2
    empty_messages()
    try:    
        file_name = file_path.get()    
        img = cv.imread(file_name)  

        params = cv.SimpleBlobDetector_Params()

        mintresh = min(minThreshold.get(),maxThreshold.get())
        maxtresh = max(minThreshold.get(),maxThreshold.get())
        minThreshold.set(mintresh)
        maxThreshold.set(maxtresh)


        params.minThreshold = minThreshold.get()
        params.maxThreshold = maxThreshold.get()
        params.thresholdStep = thresholdStep.get()
        params.filterByConvexity = filterByConvexity.get()
        params.minConvexity = minConvexity.get()

        params.filterByInertia = filterByInertia.get()
        params.minInertiaRatio = minInertiaRatio.get()

        params.filterByArea = filterByArea.get()
        params.minArea = minArea.get()
        params.maxArea = maxArea.get()

        params.filterByCircularity = filterByCircularity.get()
        params.minCircularity = minCircularity.get()

        detector = cv.SimpleBlobDetector_create(params)
        particles = detector.detect(img)
        im_with_particles = cv.drawKeypoints(img, particles, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        file_name = file_name.split('/')[-1][:-4]

        original_file_name = h1.cget("text")
        results_folder = 'Results/'+original_file_name[:-4] 

        prefix = get_unique_prefix(results_folder)


        new_file_path = results_folder+"/"+prefix+"_detected_particles_"+file_name+".png"
        metadata_path = results_folder+"/"+prefix+"_metadata_"+file_name+".csv"     
        particle_data_path = results_folder+"/"+prefix+"_particle_distribution_"+file_name+".csv"
        histogram_path = results_folder+"/"+prefix+"_histogram_"+file_name+".png"
        cv.imwrite( os.path.join(new_file_path), im_with_particles);

               
        metadata = {}
        metadata['source_file'] = file_path.get() 
        metadata['output_file'] = new_file_path
        metadata['particle_distribution_file'] = particle_data_path
        metadata['histogram_figure'] = histogram_path
        metadata['average_particle_diameter_px'] = 0
        metadata['average_particle_diameter_nm'] = '0'
        metadata['particles_detected'] = len(particles)
        metadata['minThreshold'] = minThreshold.get()
        metadata['maxThreshold'] = maxThreshold.get()
        metadata['thresholdStep'] = thresholdStep.get()
        metadata['filterByConvexity'] = filterByConvexity.get()
        metadata['minConvexity'] = minConvexity.get()
        metadata['filterByInertia'] = filterByInertia.get()
        metadata['minInertiaRatio'] = minInertiaRatio.get()
        metadata['filterByArea'] = filterByArea.get()
        metadata['minArea'] = minArea.get()
        metadata['maxArea'] = maxArea.get()
        metadata['filterByCircularity'] = filterByCircularity.get()
        metadata['minCircularity'] = minCircularity.get()

        pixelsize = str(pixel_size_label.cget("text")).split(': ')[1]

        pixelsizeunits = pixelsize.split(' ')[1]
        pixelsize = pixelsize.split(' ')[0]
        pixelsize = float(pixelsize)*unit_dict[pixelsizeunits]/unit_dict['nm']

        metadata['pixelsize'] = str(pixelsize)+' nm'



        particle_data = pd.DataFrame(columns=['coordinates','diameter (px)','size (px^2)','diameter (nm)','size (nm^2)'])
        particle_data['size'] = np.zeros(len(particles))
        for i in range(len(particle_data)):
            particle_data['coordinates'][i] = particles[i].pt
            particle_data['diameter (px)'][i] = particles[i].size
            particle_data['size (px^2)'][i] = math.pi*(particles[i].size/2)**2
            true_particle_size = pixelsize*particles[i].size
            particle_data['diameter (nm)'][i] = true_particle_size
            particle_data['size (nm^2)'][i] = math.pi*(true_particle_size/2)**2

        metadata['average_particle_diameter_px'] = str(particle_data['diameter (px)'].to_numpy().mean())+' px'
        metadata['average_particle_diameter_nm'] = str(particle_data['diameter (nm)'].to_numpy().mean())+' nm'
        metadata = pd.DataFrame.from_dict(metadata, orient='index')
        metadata.to_csv(metadata_path,header=False)

        particle_data.to_csv(particle_data_path)

        canvas2.get_tk_widget().pack_forget() 
        fig2,ax2 = plt.subplots(figsize=(3,2))
        result_message = ''
        if pixelsize > 0:
            ax2.hist((particle_data['diameter (nm)'].to_numpy()),bins=25)
            result_message = 'Average particle diameter: {:.2f}'.format((particle_data['diameter (nm)'].to_numpy().mean()))+" nm"
            ax2.set_xlabel('particle diameter (nm)')
        else:
            ax2.hist((particle_data['diameter (px)'].to_numpy()),bins=25)
            result_message = 'Average particle diameter: {:.2f}'.format((particle_data['diameter (px)'].to_numpy().mean()))+" nm"
            ax2.set_xlabel('particle diameter (px)')
            
        
        coverage_results.config(text = result_message+"\nTotal particles detected: "+str(len(particles)))
        ax2.set_ylabel('frequency')
        fig2.tight_layout()
        fig2.savefig(histogram_path, dpi=300) 
        
        canvas2 = FigureCanvasTkAgg(fig2, area_frame)
        canvas2.get_tk_widget().pack(pady = (10,0))

        canvas.get_tk_widget().pack_forget() 
        fig, ax = plt.subplots(figsize=(10,10),dpi=300)
        ax.imshow(cv.cvtColor(im_with_particles, cv.COLOR_BGR2RGB))
        ax.axis('off')
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(padx = 5,pady = (30,0))
        result_label.config(text = "Results saved in: "+new_file_path)
    except:
        l2.config(text = "No file found")
    return


def get_coverage():
    global canvas,canvas2
    empty_messages()
    try:

        file_name = file_path.get()    
        img = cv.imread(file_name) 
        im_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh = colorThreshold.get()
        im_bw = cv.threshold(im_gray, thresh, 255, cv.THRESH_BINARY)[1]
        array_counter = im_bw.reshape(-1,1).reshape(1,-1)[0]

        unique, counts = np.unique(array_counter, return_counts=True)
        coverage_dict = dict(zip(unique, counts))


        white_px = 0
        black_px = 0

        try:
            white_px = int(coverage_dict[255])
        except:
            white_px = 0

        try:
            black_px = int(coverage_dict[0])
        except:
            black_px = 0

        total_px = int(white_px+black_px)
        white_percent = float(white_px)/float(total_px)*100
        black_percent = float(black_px)/float(total_px)*100

        original_file_name = h1.cget("text")[:-4]
        results_folder = 'Results/'+original_file_name 
        prefix = get_unique_prefix(results_folder)

        file_name = file_name.split('/')[-1][:-4]

        grey_im_path = results_folder+"/"+prefix+"_greyscale_"+file_name+".png"
        bw_im_path = results_folder+"/"+prefix+"_binary_"+file_name+".png"     
        metadata_path = results_folder+"/"+prefix+"_metadata_"+file_name+".csv"

        metadata = {}
        metadata['source image'] = file_name
        metadata['greyscale image'] = grey_im_path
        metadata['binary image'] = bw_im_path 
        metadata['threshold'] = thresh
        metadata['white px'] = white_px 
        metadata['black px'] = black_px
        metadata['total_px'] = total_px
        metadata['white percent'] = white_percent
        metadata['black percent'] = black_percent

        metadata = pd.DataFrame.from_dict(metadata, orient='index')
        metadata.to_csv(metadata_path,header=False)

        cv.imwrite( os.path.join(grey_im_path), im_gray);
        cv.imwrite( os.path.join(bw_im_path), im_bw);

        coverage_results.config(text = "White: {:.3f}% \nBlack: {:.3f}%".format(white_percent,black_percent))

        canvas.get_tk_widget().pack_forget() 
        canvas2.get_tk_widget().pack_forget() 

        fig, ax = plt.subplots(figsize=(10,10),dpi=300)

        ax.imshow(cv.cvtColor(im_bw, cv.COLOR_BGR2RGB),alpha=1,cmap='gray', vmin=0, vmax=255)
        #ax.imshow(img,alpha=0.5)
        ax.axis('off')
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().pack(padx = 5,pady = (30,0))

        fig2, ax2 = plt.subplots(figsize=(3,3))    
        ax2.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax2.axis('off')
        fig2.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig2, area_frame)
        canvas2.get_tk_widget().pack(pady = (10,0))

        result_label.config(text = "Results saved in: "+bw_im_path)
    except:
        l2.config(text = "No file found")

    return


root = tk.Tk()  
root.title("Simple TEM particle detector")  
root.geometry("1200x900") 
#variables in tkinter widget
file_path = tk.StringVar()
v1 = tk.IntVar(value=79)
pixel_size = tk.DoubleVar(value = 1)
#create image field
fig, ax = plt.subplots(figsize=(10,10))
fig = fig.clear()
canvas = FigureCanvasTkAgg(fig, root)


h1 = tk.Label(root, text='',font=("Arial", 26))
h1.pack(anchor=tk.N,padx = 50,pady = (10,0))

#data loader
control_frame = tk.Frame(root,width=320)

l1 = tk.Label(control_frame, text='File loader',font=("Arial", 16))
l1_1 = tk.Label(control_frame, text='File path')
w1 = tk.Entry(control_frame, textvariable=file_path,width=50,justify=tk.LEFT)
b1 = tk.Button(control_frame, text="load image", command=load_image)
l2 = tk.Label(control_frame, text="",fg='red')


l1.pack(anchor=tk.NW)
l1.pack(anchor=tk.NW)
w1.pack(anchor=tk.NW)
b1.pack(anchor=tk.NW)
l2.pack(anchor=tk.NW)

analyze_frame = tk.Frame(control_frame,width=320)
detect_frame = tk.Frame(analyze_frame,width=160)
area_frame = tk.Frame(analyze_frame,width=160)
#preprocessing

l3 = tk.Label(control_frame, text="Preprocessing Steps",font=("Arial", 16))
b2 = tk.Button(control_frame, text="Original", command=restore_original,width=31)
b3 = tk.Button(control_frame, text="Invert Image", command=invert_image,width=31)


crop_frame = tk.Frame(control_frame,width=120)

s1 = tk.Scale(crop_frame, variable = v1, from_ = 0, to = 150, resolution=1, orient = tk.HORIZONTAL) 
e1_1 = tk.Entry(crop_frame, textvariable=v1,width=5)
b4 = tk.Button(crop_frame, text="Crop Bottom", command=crop_image)


pixel_size_label = tk.Label(control_frame, text="pixel width:",justify=tk.LEFT)

l3.pack(anchor=tk.NW,pady = (10,0))
b2.pack(anchor=tk.NW)
b3.pack(anchor=tk.NW)

s1.pack(side=tk.LEFT,pady = (0,0))
e1_1.pack(side=tk.LEFT,pady=(17,0))
b4.pack(side=tk.RIGHT,pady = (8,0),padx=(7,0))
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
maxArea = tk.IntVar(value = 1000000) 

filterByCircularity = tk.BooleanVar(value=False)
minCircularity = tk.DoubleVar(value = 0.1) 

    
l4 = tk.Label(detect_frame, text="Particle Detection",font=("Arial", 16))

threshold_frame_1 = tk.Frame(detect_frame,width=120)
threshold_frame_2 = tk.Frame(detect_frame,width=120)
threshold_frame_3 = tk.Frame(detect_frame,width=120)
l5 = tk.Label(threshold_frame_1, text="Minimal Pixel \nIntensity Threshold",justify=tk.LEFT)
l5_1 = tk.Label(threshold_frame_2, text="Maximal Pixel \nIntensity Threshold",justify=tk.LEFT)
s2 = tk.Scale(threshold_frame_1, variable = minThreshold, from_ = 0, to = 255, resolution=1, orient = tk.HORIZONTAL)  
s2_1 = tk.Scale(threshold_frame_2, variable = maxThreshold, from_ = 0, to = 255, resolution=1, orient = tk.HORIZONTAL)  
e1 = tk.Entry(threshold_frame_1, textvariable=minThreshold,width=5)
e2 = tk.Entry(threshold_frame_2, textvariable=maxThreshold,width=5)

l5_2 = tk.Label(threshold_frame_3, text="Threshold \nStep",justify=tk.LEFT)
s2_2 = tk.Scale(threshold_frame_3, variable = thresholdStep, from_ = 0, to = 5, resolution=0.5, orient = tk.HORIZONTAL)  
e2_2 = tk.Entry(threshold_frame_3, textvariable=thresholdStep,width=5)

convexity_frame_1 = tk.Frame(detect_frame,width=120)
convexity_frame_2 = tk.Frame(detect_frame,width=120)

l6 = tk.Label(convexity_frame_1, text="Filter by Convexity",justify=tk.LEFT)
r1 = tk.Checkbutton(convexity_frame_1, variable=filterByConvexity)

l7 = tk.Label(convexity_frame_2, text="Minimal \nConvexity",justify=tk.LEFT)
s3 = tk.Scale(convexity_frame_2, variable = minConvexity, from_ = 0, to = 1, resolution=0.01, orient = tk.HORIZONTAL)  
e3 = tk.Entry(convexity_frame_2, textvariable=minConvexity,width=5)

inertia_frame_1 = tk.Frame(detect_frame,width=120)
inertia_frame_2 = tk.Frame(detect_frame,width=120)
l8 = tk.Label(inertia_frame_1, text="Filter by Intertia",justify=tk.LEFT)
l9 = tk.Label(inertia_frame_2, text="Minimal Inertia\nRatio",justify=tk.LEFT)
r8 = tk.Checkbutton(inertia_frame_1, variable=filterByConvexity)
s9 = tk.Scale(inertia_frame_2, variable = minInertiaRatio, from_ = 0, to = 1, resolution=0.01, orient = tk.HORIZONTAL)  
e9 = tk.Entry(inertia_frame_2, textvariable=minInertiaRatio,width=5)

area_frame_1 = tk.Frame(detect_frame,width=120)
area_frame_2 = tk.Frame(detect_frame,width=120)
area_frame_3 = tk.Frame(detect_frame,width=120)
l10 = tk.Label(area_frame_1, text="Filter by Area",justify=tk.LEFT)
l11 = tk.Label(area_frame_2, text="Minimal Area\nSize",justify=tk.LEFT)

l11_1 = tk.Label(area_frame_3, text="Maximal Area\nSize",justify=tk.LEFT)
s11_1 = tk.Scale(area_frame_3, variable = maxArea, from_ = 0, to = 1000000, resolution=1, orient = tk.HORIZONTAL)  
e11_1 = tk.Entry(area_frame_3, textvariable=maxArea,width=10)

r10 = tk.Checkbutton(area_frame_1, variable=filterByArea)
s11 = tk.Scale(area_frame_2, variable = minArea, from_ = 0, to = 300, resolution=1, orient = tk.HORIZONTAL)  
e11 = tk.Entry(area_frame_2, textvariable=minArea,width=5)

circularity_frame_1 = tk.Frame(detect_frame,width=120)
circularity_frame_2 = tk.Frame(detect_frame,width=120)
l12 = tk.Label(circularity_frame_1, text="Filter by Circularity",justify=tk.LEFT)
l13= tk.Label(circularity_frame_2, text="Minimal\nCircularity",justify=tk.LEFT)
r12 = tk.Checkbutton(circularity_frame_1, variable=filterByCircularity)
s13 = tk.Scale(circularity_frame_2, variable = minCircularity, from_ = 0, to = 1, resolution=0.01, orient = tk.HORIZONTAL)  
e13 = tk.Entry(circularity_frame_2, textvariable=minCircularity,width=5)

       

b5 = tk.Button(detect_frame, text="Detect Particles", command=detect_particles)
result_label = tk.Label(control_frame, text="")

l4.pack(anchor=tk.NW,pady = (10,0))

l5.pack(side=tk.LEFT)
e1.pack(side=tk.RIGHT,pady=(17,0))
s2.pack(side=tk.RIGHT,padx=(1,0))
threshold_frame_1.pack(anchor=tk.NW)

l5_1.pack(side=tk.LEFT)
e2.pack(side=tk.RIGHT,pady=(17,0))
s2_1.pack(side=tk.RIGHT,padx=(1,0))
threshold_frame_2.pack(anchor=tk.NW)

l5_2.pack(side=tk.LEFT)
e2_2.pack(side=tk.RIGHT,pady=(17,0))
s2_2.pack(side=tk.RIGHT,padx=(46,0))
threshold_frame_3.pack(anchor=tk.NW)

l6.pack(side=tk.LEFT)
r1.pack(side=tk.RIGHT,padx=(1,0))
convexity_frame_1.pack(anchor=tk.NW)

l7.pack(side=tk.LEFT)
e3.pack(side=tk.RIGHT,pady=(17,0))
s3.pack(side=tk.RIGHT,padx=(48,0))
convexity_frame_2.pack(anchor=tk.NW)


l8.pack(side=tk.LEFT)
r8.pack(side=tk.RIGHT,padx=(17,0))
inertia_frame_1.pack(anchor=tk.NW)

l9.pack(side=tk.LEFT)
e9.pack(side=tk.RIGHT,pady=(17,0))
s9.pack(side=tk.RIGHT,padx=(21,0))
inertia_frame_2.pack(anchor=tk.NW)

l10.pack(side=tk.LEFT)
r10.pack(side=tk.RIGHT,padx=(30,0))
area_frame_1.pack(anchor=tk.NW)

l11.pack(side=tk.LEFT)
e11.pack(side=tk.RIGHT,pady=(17,0))
s11.pack(side=tk.RIGHT,padx=(30,0))
area_frame_2.pack(anchor=tk.NW)

l11_1.pack(side=tk.LEFT)
e11_1.pack(side=tk.RIGHT,pady=(17,0))
s11_1.pack(side=tk.RIGHT,padx=(28,0))
area_frame_3.pack(anchor=tk.NW)

l12.pack(side=tk.LEFT)
r12.pack(side=tk.RIGHT,padx=(0,0))
circularity_frame_1.pack(anchor=tk.NW)

l13.pack(side=tk.LEFT)
e13.pack(side=tk.RIGHT,pady=(17,0))
s13.pack(side=tk.RIGHT,padx=(47,0))
circularity_frame_2.pack(anchor=tk.NW)

b5.pack(anchor=tk.NW,pady=(10,0))


fig2, ax2 = plt.subplots(figsize=(0.2,0.2),dpi=72)
fig2 = fig2.clear()
canvas2 = FigureCanvasTkAgg(fig2, control_frame)


l16 = tk.Label(area_frame, text="Coverage Calculator",font=("Arial", 16))

colorThreshold = tk.IntVar(value = 170)
color_threshold_frame = tk.Frame(area_frame,width=120)

l17 = tk.Label(color_threshold_frame, text="Separate on Pixel \nIntensity Threshold",justify=tk.LEFT)
s17 = tk.Scale(color_threshold_frame, variable = colorThreshold, from_ = 0, to = 255, resolution=1, orient = tk.HORIZONTAL)  
e17 = tk.Entry(color_threshold_frame, textvariable=colorThreshold,width=5)

b17 = tk.Button(area_frame, text="Analyze Coverage", command=get_coverage)

coverage_results = tk.Label(area_frame, text="\n",justify=tk.LEFT) 
# get area frame + histogram

l16.pack(anchor=tk.NW,pady = (10,0))

l17.pack(side=tk.LEFT)
e17.pack(side=tk.RIGHT,pady=(17,0))
s17.pack(side=tk.RIGHT,padx=(1,0))
color_threshold_frame.pack(anchor=tk.NW)
b17.pack(anchor=tk.NW,pady=(10,0))

coverage_results.pack(anchor=tk.NW,pady=(10,0))

detect_frame.pack(side=tk.LEFT,anchor=tk.NW)
area_frame.pack(side=tk.RIGHT,anchor=tk.NW,padx=15)
analyze_frame.pack()

result_label.pack(anchor=tk.NW)

control_frame.pack(side=tk.LEFT,anchor=tk.NW,padx = (10,5),pady = (0,0),expand=False)
canvas.get_tk_widget().pack(side=tk.RIGHT,padx = 20,pady = (30,0)) 


root.mainloop()