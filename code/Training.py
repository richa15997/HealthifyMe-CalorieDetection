import os
import pip
import sys
import shutil
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

def model_paths():
    vars={
    'CUSTOM_MODEL_NAME' :'my_ssd_mobnet' ,
    #'CUSTOM_MODEL_NAME':'my_mask_rcnn',
    'PRETRAINED_MODEL_NAME' :'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
    'PRETRAINED_MODEL_URL' : 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
    #'PRETRAINED_MODEL_NAME' :'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8',
    #'PRETRAINED_MODEL_URL':'http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz',
    'TF_RECORD_SCRIPT_NAME' :'generate_tfrecord.py',
    'LABEL_MAP_NAME' : 'label_map.pbtxt'}

    paths = {
    'WORKSPACE_PATH': os.path.join('code','Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('code','Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('code','Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('code','Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('code','Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('code','Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('code','Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('code','Tensorflow', 'workspace','models',vars['CUSTOM_MODEL_NAME']), 
    'OUTPUT_PATH': os.path.join('code','Tensorflow', 'workspace','models',vars['CUSTOM_MODEL_NAME'], 'export'), 
    'TFJS_PATH':os.path.join('code','Tensorflow', 'workspace','models',vars['CUSTOM_MODEL_NAME'], 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('code','Tensorflow', 'workspace','models',vars['CUSTOM_MODEL_NAME'], 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('code','Tensorflow','protoc'),
    'TF_RECORD_OID':os.path.join('code','Tensorflow','tf_record_git')
    }

    files = {
        'PIPELINE_CONFIG':os.path.join('code','Tensorflow', 'workspace','models', vars['CUSTOM_MODEL_NAME'], 'pipeline.config'),
        #'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], vars['TF_RECORD_SCRIPT_NAME']), 
        'TF_RECORD_SCRIPT':os.path.join(paths['TF_RECORD_OID'],vars['TF_RECORD_SCRIPT_NAME']),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], vars['LABEL_MAP_NAME'])
    }

    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
            
    return paths,files,vars

def download_objdetection_modules(root_path,paths):
        
        os.system('brew install wget') #wget is a free tool scrape data from websites
        
        #cloning tensorflow model garden
        dest_path=os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')
        clone="git clone https://github.com/tensorflow/models"
        if not os.path.exists(dest_path):
            #print(os.path.join(os.getcwd(),'code/Tensorflow'))
            os.chdir(os.path.join(root_path,'code/Tensorflow'))
            #print(os.getcwd())
            os.system(clone)
        else:
            print("Models are already cloned!")
        
        # # Install Tensorflow Object Detection 
        if os.name=='posix':  
            os.system('brew install protobuf')
            os.system('brew link protobuf')
            os.system('cd code/Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . --user') 
            #have to write script to add to path

def verification_script(paths): #is the verification script to check that all object detection modules have been downloaded
    VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
    script='python '+VERIFICATION_SCRIPT
    os.system(script)

def download_pretrain_models(root_path,paths,vars):
    os.chdir(os.path.join(root_path,paths['PRETRAINED_MODEL_PATH']))
    if not os.path.exists(vars['PRETRAINED_MODEL_NAME']+'.tar.gz'):
        os.system('wget '+ vars['PRETRAINED_MODEL_URL'])
        shutil.move(vars['PRETRAINED_MODEL_NAME']+'.tar.gz', paths['PRETRAINED_MODEL_PATH'])
        os.system('cd '+paths['PRETRAINED_MODEL_PATH']+' && tar -zxvf '+vars['PRETRAINED_MODEL_NAME']+'.tar.gz')

def create_label_map(root_path,files):
    labels=[{'name':'Apple','id':1},{'name':'Orange','id':2},{'name':'Banana','id':3},{'name':'Pear','id':4},{'name':'Watermelon','id':5},{'name':'Strawberry','id':6}]
    os.chdir(root_path)
    with open(files["LABELMAP"],"w") as file:
        for label in labels:
            file.write('item { \n')
            file.write('\tname:\'{}\'\n'.format(label['name']))
            file.write('\tid:{}\n'.format(label['id']))
            file.write('}\n')
    print("Label maps created!")
    return labels

def generate_tfrecord(root_path,files,paths):
    #below is code for cloning TF Record scripts in folder 
    os.chdir(os.path.join(root_path,paths['TF_RECORD_OID']))
    
    if not os.path.exists(os.getcwd()):
            clone="git clone https://github.com/zamblauskas/oidv4-toolkit-tfrecord-generator"
            os.system(clone)
            print("Cloned TF Record repository!")
    else:
        print("TF Record already cloned!")
    
    os.chdir(os.path.join(os.getcwd(),"oidv4-toolkit-tfrecord-generator"))
    
    #below is code for converting OIDv4 train data to TF Record format
    
    script="python generate-tfrecord.py \
    --classes_file=../../../../OIDv4_ToolKit/classes.txt \
    --class_descriptions_file=../../../../OIDv4_ToolKit/OID/csv_folder/class-descriptions-boxable.csv \
    --annotations_file=../../../../OIDv4_ToolKit/OID/csv_folder/train-annotations-bbox.csv \
    --images_dir=../../../../OIDv4_ToolKit/OID/Dataset/train_downsample \
    --output_file=../../workspace/annotations/train.tfrecord"
    os.system(script)
    
    #below is code for converting OIDv4 test data to TF Record format
    
    script="python generate-tfrecord.py \
    --classes_file=../../../../OIDv4_ToolKit/classes.txt \
    --class_descriptions_file=../../../../OIDv4_ToolKit/OID/csv_folder/class-descriptions-boxable.csv \
    --annotations_file=../../../../OIDv4_ToolKit/OID/csv_folder/test-annotations-bbox.csv \
    --images_dir=../../../../OIDv4_ToolKit/OID/Dataset/test_downsample \
    --output_file=../../workspace/annotations/test.tfrecord"
    os.system(script)
    

def copy_config_file(root_path,vars,paths):
    #copy the pipeline cofig file from pre-trained model to model folder
    os.chdir(root_path)
    #print(os.path.join(root_path,paths["PRETRAINED_MODEL_PATH"],vars["PRETRAINED_MODEL_NAME"]))
    shutil.copy(os.path.join(root_path,paths["PRETRAINED_MODEL_PATH"],vars["PRETRAINED_MODEL_NAME"],'pipeline.config'),paths["CHECKPOINT_PATH"])
    #shutil.copy(os.path.join(paths["PRETRAINED_MODEL_PATH"],vars["PRETRAINED_MODEL_NAME"],'pipeline.config'),paths["CHECKPOINT_PATH"])
    print("Model config file copied!")

def update_config_file(root_path,vars,files,paths,labels):
    #print(os.system("pip list"))
    #print(sys.path)
    import tensorflow as tf
    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format
    #print(files['PIPELINE_CONFIG'])
    #config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    #print(config)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    # #copy the original config file to empty pipeline_config 
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)
    
    #configuration files are updated
    pipeline_config.model.ssd.num_classes = len(labels)
    #pipeline_config.train_config.batch_size = 64
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], vars['PRETRAINED_MODEL_NAME'], 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.tfrecord')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.tfrecord')]
    #print(pipeline_config)

    #update the original config file to pipeline config
    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text)
    

def train_model(paths,files):
    print(os.getcwd())
    TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])
    os.system("python3 -m  pip uninstall pycocotools -y")
    os.system("python3 -m  pip install pycocotools==2.0.0")
    os.system(command)
     

def eval_model(paths,files):
    print(os.getcwd())
    print(paths['CHECKPOINT_PATH'])
    TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
    command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={} ".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
    os.system(command)
    
def visualize_train_results(paths,files):
    os.chdir(os.path.join(paths["CHECKPOINT_PATH"],"train"))
    os.system("python "+paths["CHECKPOINT_PATH"]+"/main.py"+" --logdir=.")

def visualize_eval_results(paths,files):
    os.chdir(os.path.join(paths["CHECKPOINT_PATH"],"eval"))
    print(os.getcwd())
    # os.system("pip show tensorflow")
    # os.system("cd tensorflow")
    os.system("python "+os.path(paths["CHECKPOINT_PATH"],"main.py")+" --logdir=.")
    #/Users/richaranderia/Documents/HealthifyMe_Project_Detection/code/Tensorflow/workspace/models/my_ssd_mobnet/eval/main.py
    #os.system("tensorboard --logdir=.")

def load_train_model_from_checkpoint(root_path,paths,files):
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    # Restore checkpoint
    # ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    # ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-5')).expect_partial()
    tf.saved_model.save(detection_model, os.path.join(root_path,"./final_model"))
    #tf.keras.models.(detection_model,os.path.join(root_path,"code/final_model"),save_format='tf')
def install_packages():
    root_path=os.getcwd()
    #os.system('python3 -m pip install -r '+os.path.join("code","requirements.txt"))
    #os.system(packages)
    paths,files,vars=model_paths() #initialize all paths
    #download_objdetection_modules(root_path,paths)#download pre trained models from tensorflow zoo
    #verification_script(paths)
    #download pretrained models
    #download_pretrain_models(root_path,paths,vars)
    #create label map that is apple=1 etc
    labels=create_label_map(root_path,files)
    #create tfrecord files
    #generate_tfrecord(root_path,files,paths)
    #copy the model configurations to training folder
    #copy_config_file(root_path,vars,paths)
    #update the model architecture
    #update_config_file(root_path,vars,files,paths,labels)
    #train the model
    train_model(paths,files)
    #visualize_train_results(paths,files)
    #test the model
    # eval_model(paths,files)
    # visualize_eval_results(paths,files)
    #load_train_model_from_checkpoint(root_path,paths,files)
    

if __name__=="__main__":
    install_packages()