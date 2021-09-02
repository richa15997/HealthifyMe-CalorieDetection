import random
import os
import shutil

def downsample(division,num): #Downsample training images

    fruits=['Apple','Banana','Orange','Pear','Strawberry','Watermelon']
    for fruit in fruits:
        images=[]
        for root,dirs,files in os.walk(division+"/"+fruit):
            for file in files:
                if(file.split(".")[1]=="jpg"):        
                    images.append(file)

        sample=random.sample(images,num)

        if(os.path.isdir(division+"_downsample/"+fruit)):
            shutil.rmtree(division+"_downsample/"+fruit)   
        if(os.path.isdir(division+"_downsample/"+fruit+"/Label")):
            shutil.rmtree(division+"_downsample/"+fruit+"/Label")
        os.makedirs(division+"_downsample/"+fruit)
        os.makedirs(os.path.join(division+"_downsample/"+fruit,"Label"))
        for img_id in sample:
            shutil.copy(os.path.join(division+"/"+fruit,img_id),division+"_downsample/"+fruit)
            shutil.copy(os.path.join(division+"/"+fruit+"/Label",img_id.split('.')[0]+'.txt'),division+"_downsample/"+fruit+"/Label")
        #print(len(apple_sample))

if __name__=="__main__":
    os.chdir(os.path.join("OIDv4_Toolkit","OID","Dataset"))
    downsample("train",200)
    downsample("test",50)
    