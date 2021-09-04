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
The project was implimented in Python (version 3.9.6) that uses Tensorflow object detection models (2.5.0) and tested on ubuntu and Windows OS. Dataset of food items was collected from Open Images Dataset and downloaded using the [OIDV4 ToolKit](https://github.com/EscVM/OIDv4_ToolKit).

## Run Demo
Download all the folders and execute the predictor_flask.py script using format -> ```gunicorn -p 0.0.0.0.8080 predictor_flask:app```. Note 8080 is the port chosen by me, you can use other available ports as well.

## Overview
Food recognition was performed using transfer learning of SSD MobileNet model. The SSD Mobilenet model, aka Single Shot Multibox Detector is used to perform localization ie. identifying bounding boxes around objects and MobileNet model is used for classification ie. distinguishing which object is present. The combination of SSD and Mobilenet models priduces an object detection model. 
The SSD model is based on the VGG16 architecture which extracts feature maps and the latter top Conv4_3 layers allows detection of objects. 
![1*aex5im2aYcsk4RVKUD4zeg](https://user-images.githubusercontent.com/28618175/131443439-d3661280-7bbd-4df6-80d1-fefd69e4c2f5.jpeg)

