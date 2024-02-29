The widget is made to ease the use of the OpenCV SimpleBlobDetector algorithm.
To use the widget for those new to python:
1. Install Anaconda https://www.anaconda.com/
2. Install the dependent packages:
  Open the anaconda terminal and run:
    conda install matplotlib
    pip install opencv-python
    pip install pillow

To run the widget:
1. Open anaconda terminal
2. cd /d "directory where particle_detection.py is located"
3. python particle_detection.py

Or if conda is the default python interpreter in the PATH variables
1. The quickstart calls the default anaconda environment and then launches the python script

Notes on opening files:
1. Put the files in the same folder as particle_detection.py
2. Type the name of the file with extension in the file loader
3. You can open a previous file from the results folder in the same way: Results/sample/0000_original_sample.tif

Notes on preprocessing:
- The algorithm detects black particles, thus there is an inverse button if necessary.
- The enhanced contrast normalizes the colors in the figure to span the entire range. It sets the darkest color to 0 and the lightest to 255.

Notes on data storage/generation:
- Each edit is stored in the Results/sample folder and does not/should not overwrite previous files
- When detecting particles there are two data sets generated: keypoints and particles. Keypoints detect areas of interest and assume perfect spheres. Particles detect contours of all kind of shapes.
- The metadata csv contains the important data for reproducibility and relating files to each other
- The preview histogram shows the diameter from true area

Data stored about keypoints:
  diameter - the diameter of the keypoint
  size - the area of the keypoint assuming a sphere


Data stored about particles:
  perfect circle diameter - calculated from the minimal circle that encapsulates the entire detected particle. Larger than the true size
  perfect circle size - the area of the minimal circle that encapsulates the entire detected particle. Larger than the true size
  contour perimiter - the perimiter of the nanoparticle
  true area of particle - the true area of enclosed by the detected contour
  diameter from true area - the diameter calculated from the true area assuming a perfect sphericality
  sphericality - the sphericality of the detected particle where 1 is a perfect circle

When using please cite the package as:
Mints, V. A. (2024). easy_nanoparticle_detector. GitHub. https://github.com/vamints/easy_nanoparticle_detector
