import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
os.chdir(os.path.join("OIDv4_Toolkit","OID","Dataset","train"))
watermelon_images=[]
for root,dirs,files in os.walk("Watermelon"):
        for file in files:
                if(file.split(".")[1]=="jpg"):        
                        watermelon_images.append(file)
        
for image in watermelon_images:
        print(image)
        print("done")
        img = load_img(os.path.join("Watermelon",image)) 
        x = img_to_array(img)  
        x = x.reshape((1,) + x.shape)  
        #print(x.shape)
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir=os.path.join('watermelon_upsample'), save_format='jpeg'):
                if i > 2:#number of augmented images for eaach image
                        break  
                i+=1