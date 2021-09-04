# HealthifyMe-CalorieDetection
"Work smart before you start working hard", same applies to all those determined to improve their lifestyle. This is a health based application which will assist users to determine the amount of calories present in their food depending on the size of the item. The application has been deployed using the Flask framework on the GCP Server (earlier). The flask application can now be downloaded on your local machine and launched on your localhost using the gunicorn server. Users just need to choose a file and upload it on the link.

<img width="800" alt="image" src="https://user-images.githubusercontent.com/28618175/131414809-5387b16c-ff31-4187-ba74-c47361bcf18c.png">
Above image is the intro page
<img width="800" alt="image" src="https://user-images.githubusercontent.com/28618175/131423677-ef6b752d-1267-4206-b68c-58d7fe65df43.png">
Above image is the page to upload an image of your choice
<img width="800" alt="image" src="https://user-images.githubusercontent.com/28618175/131423709-09ecf899-e011-4fce-83f4-54863b49a469.png">
Above image is the page to view the image that you chose. Once you select the "predict" button the food item will be determined and its calories will be returned.
<img width="800" alt="image" src="https://user-images.githubusercontent.com/28618175/131424237-8e180f4c-557a-4c32-a230-a6239c48d6cc.png">
Above image is the calories predicted of the given image based on the size of the item.

## Prerequisites
The project was implimented in Python (version 3.9.6) that uses Tensorflow object detection models (2.5.0) and tested on ubuntu and Windows OS. 

## Run Demo
Download all the folders and execute the predictor_flask.py script using format -> ```gunicorn -p 0.0.0.0.8080 predictor_flask:app```. Note 8080 is the port chosen by me, you can use other available ports as well.

## Overview
Food recognition was performed using transfer learning of SSD MobileNet model. The combination of SSD and Mobilenet model produces an object detection detection model. The SSD MobileNet model consists of two parts; the base network and the detection network. 

Mobilenet is the base network which is used for object recognition. 
<img width="1314" alt="image" src="https://user-images.githubusercontent.com/28618175/132108405-9a2aa2dc-6f4d-4d27-81a7-c04648120cb0.png">

The SSD is the detection network which ensures multiple objects can be detected from a single shot image. Therefore the SSD network can be placed on different base networks such as ResNet, MobileNet, R-CNN, Yolo etc. 

<img width="925" alt="image" src="https://user-images.githubusercontent.com/28618175/132108297-93977f87-d9fc-4d38-9352-ccf990e724c2.png">

The choice of MobileNet as the base network on the SSD model is beneficial as both seek to perform object recognition in computationally limited devices. Therefore the combination of both provides high accuracy tradeoff and fast computational results in real-time. 

<img width="782" alt="image" src="https://user-images.githubusercontent.com/28618175/132108417-fd6a15e7-355a-4964-b26f-f62a1c5a9faf.png">

Dataset of food items was collected from [Open Images Dataset](https://g.co/dataset/open-images) and downloaded using the [OIDV4 ToolKit](https://github.com/EscVM/OIDv4_ToolKit).

The training code can be found in script code/Training.py

## Working
The dockerfile contains the dockerized version of the code whose image can be downloaded and run on your personal PC. It is also possible to run the application without downloading the dockerized image by downloading predictor_flask.py The HTML linked to the flask application can be found in the Templates folder. 

## References:
**SSD MobileNet understanding:** https://www.xailient.com/post/real-time-vehicle-detection-with-mobilenet-ssd-and-xailient
**Download data using OIDv4 ToolKit:** https://www.youtube.com/watch?v=dLSFX6Jq-F0&t=612s
