from flask import Flask, request, redirect, render_template
from werkzeug.utils import  secure_filename
import os
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util


vars={'LABEL_MAP_NAME' : 'label_map.pbtxt',
    'CUSTOM_MODEL_NAME' :'my_ssd_mobnet' }
paths = {'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations')}
files = {'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], vars['LABEL_MAP_NAME']),
        'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', vars['CUSTOM_MODEL_NAME'], 'pipeline.config')}

food_classes=["Apple","Orange","Banana","Pumpkin","Watermelon","Lemon"]
#all images to be tested should be saved in the static folder
app = Flask(__name__, static_url_path='/static')
#fields for flask app
app.config["IMAGE_UPLOADS"] = './static'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]

#building the model from pipeline config
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
#food_prediction_model = tf.keras.models.load_model('./final_model')
#@tf.function
def detect_fn(image):
    #detection_model=tf.saved_model.load("../final_model")
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def allowed_image(filename):
    #if filename does not have a . means extension is not defined
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]  

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True

    else:
        return False


@app.route("/", methods=["GET", "POST"]) #post is used for posting on to a website
def upload_image() :

    if request.method == "POST":

        if request.files:

            image = request.files["image"] #will take the image from the file name image in the html file
            #if uploaded file has empty file name
            if image.filename == "":
                return redirect(request.url)

            #if image is a valid image
            if allowed_image(image.filename):

                filename = secure_filename(image.filename)
                #image will be saved in static folder
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                return redirect(f'/showing-image/{filename}')

            else:
                return redirect(request.url)
    #this is an html template of how website looks like
    #print(os.getcwd())
    return render_template("upload_images.html")

@app.route("/showing-image/<image_name>", methods=["GET", "POST"])
def showing_image(image_name):
    print("hellooo",os.getcwd())
    if request.method == 'POST':
        image_path = os.path.join(app.config["IMAGE_UPLOADS"], image_name)
        image = cv2.imread(image_path) #BGR
        image=cv2.resize(image,(320,320))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        
        detections = detect_fn(input_tensor)
        #print(detections)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        
        # detections['num_detections'] = num_detections

        # # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)[0]
        food_classes=["Apple","Orange","Banana","Pumpkin","Watermelon","Lemon"]
        calories=["95 calories","62 calories","105 calories","50 calories","46 calories","20 calories"]
        # print(detections['detection_classes'])
        # label_id_offset = 1
        # image_np_with_detections = image_np.copy()
        # category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

        # viz_utils.visualize_boxes_and_labels_on_image_array(
        #             image_np_with_detections,
        #             detections['detection_boxes'],
        #             detections['detection_classes']+label_id_offset,
        #             detections['detection_scores'],
        #             category_index,
        #             use_normalized_coordinates=True,
        #             max_boxes_to_draw=5,
        #             min_score_thresh=.8,
        #             agnostic_mode=False)

        # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        # plt.show()

        return render_template("prediction_result.html",image_name=image_name,predicted_class=food_classes[detections['detection_classes']],cal=calories[detections['detection_classes']])
       
    return render_template("showing_image.html", value=image_name)

if __name__ == '__main__':
    
    
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
