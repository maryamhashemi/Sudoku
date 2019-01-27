import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

class KNN:
    
    def resize_image(self, src_image, dst_image_height = 32, dst_image_width = 32):
        src_image_height = src_image.shape[0]
        src_image_width = src_image.shape[1]
    
        if src_image_height > dst_image_height or src_image_width > dst_image_width:
            height_scale = dst_image_height / src_image_height
            width_scale = dst_image_width / src_image_width
            scale = min(height_scale, width_scale)
            img = cv2.resize(src=src_image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            img = src_image
    
        img_height = img.shape[0]
        img_width = img.shape[1]
    
        dst_image = np.ones(shape=[dst_image_height, dst_image_width], dtype=np.uint8)
        dst_image = np.multiply(dst_image, 255)
    
        y_offset = (dst_image_height - img_height) // 2
        x_offset = (dst_image_width - img_width) // 2
    
        dst_image[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = img
    
        return dst_image  

    def read_train_data(self,images_height=32, images_width=32):
        
        # Find number of train images 
        ImagesCount = 0
        for NumberClass in range(1,10):
            File_Path = './NumberDataset/Train_Data/' + str(NumberClass) + '/*.jpg'
            ImagesCount = ImagesCount + len(glob.glob(File_Path))
            File_Path = './NumberDataset/Train_Data/' + str(NumberClass) + '/*.png'
            ImagesCount = ImagesCount + len(glob.glob(File_Path))
        
        # Train images.    
        train_images= np.zeros(shape =(ImagesCount,images_height,images_width) ,dtype=np.float32)
        
        # Train labels.
        train_labels= np.zeros(shape = (ImagesCount),dtype=np.int)
    
        i=0;
        
        for NumberClass in range(1,10):
            File_Path = './NumberDataset/Train_Data/' + str(NumberClass) + '/*.jpg'
            jpgimages = ['/'.join(file.split('\\')) for file in glob.glob(File_Path)]
            
            File_Path = './NumberDataset/Train_Data/' + str(NumberClass) + '/*.png'
            pngimages = ['/'.join(file.split('\\')) for file in glob.glob(File_Path)]
            
            images =  jpgimages + pngimages   
             
            for image in images:
                # Image reading
                image = cv2.imread(image , 0)
                
                # Image resizing.
                image = self.resize_image(src_image=image, dst_image_height=images_height, dst_image_width=images_width)
                
                #Image binarization.
                image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                            cv2.THRESH_BINARY_INV,9,20)
                
                # Image.
                train_images[i] = image
    
                # Label.
                train_labels[i] = NumberClass
                
                #plt.imshow(image,cmap='gray')
                #plt.show()
                i = i+1
        
        return train_images,train_labels
    
    def read_test_data(self, images_height=32, images_width=32):
        
    # Find number of train images 
        ImagesCount = 0
        for NumberClass in range(1,10):
            File_Path = './NumberDataset/Test_Data/' + str(NumberClass) + '/*.jpg'
            ImagesCount = ImagesCount + len(glob.glob(File_Path))
            File_Path = './NumberDataset/Test_Data/' + str(NumberClass) + '/*.png'
            ImagesCount = ImagesCount + len(glob.glob(File_Path))
        
        # Train images.    
        test_images= np.zeros(shape =(ImagesCount,images_height,images_width) ,dtype=np.float32)
        
        # Train labels.
        test_labels= np.zeros(shape = (ImagesCount),dtype=np.int)
    
        i=0;
        
        for NumberClass in range(1,10):
            File_Path = './NumberDataset/Test_Data/' + str(NumberClass) + '/*.jpg'
            jpgimages = ['/'.join(file.split('\\')) for file in glob.glob(File_Path)]
            
            File_Path = './NumberDataset/Test_Data/' + str(NumberClass) + '/*.png'
            pngimages = ['/'.join(file.split('\\')) for file in glob.glob(File_Path)]
            
            images =  jpgimages + pngimages   
             
            for image in images:
                # Image reading
                image = cv2.imread(image , 0)
                
                # Image resizing.
                image = self.resize_image(src_image=image, dst_image_height=images_height, dst_image_width=images_width)
                
                # Image binarization.
                image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                              cv2.THRESH_BINARY_INV,9,20)
                
                # Image.
                test_images[i] = image
    
                # Label.
                test_labels[i] = NumberClass
                
                i = i+1
        
        return test_images,test_labels

    def PreProcessingForKNN(self,image): 
               
        #Convert the image to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        #plt.imshow(image,cmap='gray')
        #plt.show()
        
        # Image resizing.
        image = self.resize_image(src_image=image)
                
        # Image binarization.
        image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                            cv2.THRESH_BINARY_INV,9,20)
        
        #plt.imshow(image,cmap='gray')
        #plt.show()
        return image
       
    def feature_extractor(self, train_images):
        winSize = (32,32)
        blockSize = (14,14)
        blockStride = (6,6)
        cellSize = (14,14)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradients = True
     
        hog = cv2.HOGDescriptor(winSize,blockSize,
                                blockStride,cellSize,
                                nbins,derivAperture,
                                winSigma,histogramNormType,
                                L2HysThreshold,gammaCorrection,nlevels,
                                signedGradients)
        descriptor = []
        
        for im in train_images:
            
            im = np.uint8(im)
            h=hog.compute(im)
            h = h.flatten()
            descriptor.append(h)
            
        return np.float32(descriptor) 

    def training(self,features, train_labels):
        
        '''
            Set up KNN for OpenCV 3
        '''
        knn = cv2.ml.KNearest_create()
        
        '''
            Train KNN on training data
        '''
        knn.train(features, cv2.ml.ROW_SAMPLE, train_labels)
     
        return knn   
    
    def testing(self,knn,test_data):
        
        ret,result,neighbours,dist = knn.findNearest(test_data,k=3)
        return result

    def evalute(self,test_labels,prediction_label): 
        
        confusion_matrix = np.zeros((9,9),dtype = 'int')  
                
        for i in range(0,len(prediction_label)): 
            for j in range(0,9):      
                if (prediction_label[i] == j + 1):
                    confusion_matrix[test_labels[i] - 1][j] += 1
         
        mean_accuracy =0.0       
        for i in range(0,9):
            Sum = 0
            for j in range(0,9):
                Sum = Sum + confusion_matrix[i][j]
                          
            accuracy = confusion_matrix[i][i] / Sum
            mean_accuracy = mean_accuracy + accuracy
        
        mean_accuracy = mean_accuracy / 9
         
        print("confusion matrix = ")
        print(confusion_matrix)  
        print("mean accuracy = ", mean_accuracy)

    def GetTrainedModel(self):
        
        knn = KNN()
        print('Reading Train Data ...')
        train_images, train_labels = knn.read_train_data()
        
        print('Reading Test Data ...')
        test_images, test_labels = knn.read_test_data()
        
        print('Extracting hog feature from training samples ...')
        hog_features = knn.feature_extractor(train_images)
        
        print('Training with KNN ...')
        model = knn.training(hog_features, train_labels)
        
        print('Extracting hog feature from testing samples ...')
        hog_features = knn.feature_extractor(test_images)
        
        print('Testing ...')
        prediction_label = knn.testing(model, hog_features)
        
        print('Evaluating ...')
        self.evalute(test_labels, prediction_label)
        
        return model


