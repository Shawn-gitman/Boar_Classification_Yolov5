# -*- coding: utf-8 -*-
"""Boar_Classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1srx0xl4lk-GyL9jiCvrVUU3dmfHh4SJu

# Image Based Wild Boar Species Recognition using YOLOv5x 🐗



## ● What is YOLOV5?

YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development(-ref. by Ultralytics).  

* Explanation Video KR @ https://b-flask.shawngitman.repl.co/try (👈 Click!)


## ● How to Implement?

1. Save a copy of our google colaboratory in google drive.
2. Create file "YOLO"("/content/drive/My Drive/YOLO").
3. Upload Detect.py, best_original.pt, and "멧돼지데이터.zip" in "YOLO".
4. Start google colaboratory by "Run All(Runtime-Run All)".

* See Demo @ https://b-flask.shawngitman.repl.co/try (👈 Click!)

## ● How can I download the results?

* Download "/content/res.zip"
* Source available @ https://github.com/Shawn-gitman/Boar_Classification_Yolov5

## ● Where should I contact?

* 24/7 anytime reach us to taegue52@daum.net, Taekyu Kang
* Source available @ https://github.com/Shawn-gitman/Boar_Classification_Yolov5


## ● Dependencies

* Cython
* matplotlib
* numpy
* opencv-python
* Pillow
* PyYAML
* scipy
* tensorboard
* torch
* torchvision
* tqdm
* seaborn
* pandas
* coremltools
* vonnx>=1.8.1
* scikit-learn
* thop
* pycocotools>=2.0

## ● Reference

* Ultralytics
* Roboflowai

## ● Update

* 2/12/2022 - Several bugs revised & Github published completed.

# 1. Download Dependencies
"""

# Commented out IPython magic to ensure Python compatibility.
# clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5  # clone repo
# %cd yolov5
!git reset --hard 886f1c03d839575afecb059accf74296fad395b6

# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch

from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

"""# 2. Mount Google Drive & Prepare Boar Dataset"""

#mount google drive
#test: unzipped file
from google.colab import drive
drive.mount('/content/drive')
!unzip "/content/drive/My Drive/YOLO/멧돼지데이터.zip" -d "/content/test" #unzip boar image folder

#upgrade Pillow
#make Sure to "RESTART RUNTIME"
!pip install --upgrade Pillow

#create extra files
#test2: image processed files
#test3: result(excel + detected wild boar images)
!mkdir "/content/test2"
!mkdir "/content/test3"

"""# 3. Image Processing: Resize the Image"""

#image Processing: original to 416 * 416 pixel image
import glob
from IPython.display import Image, display
from PIL import Image

i = 0
for imageName in glob.glob('/content/test/*.JPG'): #assuming JPG
    name = imageName[14:]
    image = Image.open(imageName)
    new_image = image.resize((416, 416))
    new_image.save('/content/test2/' + name)
    print(str(i) +": Image resize completed.")
    i+=1

"""# 4. Replace Detect.py"""

# Commented out IPython magic to ensure Python compatibility.
#replace detect.py
!rm /content/yolov5/detect.py
# %cp "/content/drive/My Drive/YOLO/detect.py" /content/yolov5

"""# 5. Run Detect.py & Start Classifying"""

#download xlsxwriter
!pip install xlsxwriter

# Commented out IPython magic to ensure Python compatibility.
#detect & run Pretrained Model
# %cd /content/yolov5/
!python detect.py --weights "../drive/My Drive/YOLO/best_original.pt" --img 416 --conf 0.4 --source ../test2

"""# 6. Visualize the Results"""

#visualize the results

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.JPG'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")

# Commented out IPython magic to ensure Python compatibility.
#zip Results
# %cp /content/yolov5/Example2.xlsx /content/test3
!zip -r /content/res.zip /content/test3

"""# 7. Export to Google Drive"""

# Commented out IPython magic to ensure Python compatibility.
#Export res.zip to your google drive
# %cp /content/res.zip "/content/drive/My Drive/YOLO"