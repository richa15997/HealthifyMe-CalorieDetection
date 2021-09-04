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
https://static.wixstatic.com/media/b90216_85fe4f7a9d034a8da07f4710dce9a108~mv2.png/v1/fill/w_1060,h_305,al_c,lg_1/b90216_85fe4f7a9d034a8da07f4710dce9a108~mv2.png![image](https://user-images.githubusercontent.com/28618175/132108269-b4344297-14b1-4bce-94b6-c12fb628bf25.png)

The SSD is the detection network which ensures multiple objects can be detected from a single shot image. Therefore the SSD network can be placed on different base networks such as ResNet, MobileNet, R-CNN, Yolo etc. 
<img width="925" alt="image" src="https://user-images.githubusercontent.com/28618175/132108297-93977f87-d9fc-4d38-9352-ccf990e724c2.png">

The choice of MobileNet as the base network on the SSD model is beneficial as both seek to perform object recognition in computationally limited devices. Therefore the combination of both provides high accuracy tradeoff and fast computational results. 
https://static.wixstatic.com/media/b90216_caa66cfb7cd246098cac145b2faa3485~mv2.png/v1/fill/w_1256,h_468,al_c/b90216_caa66cfb7cd246098cac145b2faa3485~mv2.png![image](https://user-images.githubusercontent.com/28618175/132108334-36d8a7aa-e904-49ed-abaa-05bd131bd2a6.png)

Dataset of food items was collected from Open Images Dataset and downloaded using the [OIDV4 ToolKit](https://github.com/EscVM/OIDv4_ToolKit).
