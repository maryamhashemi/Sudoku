import cv2
from matplotlib import pyplot as plt
import numpy as np
from blaze.expr.strings import isdigit

'''
    Sort corners: top-left, bot-left, bot-right, top-right 
    Inspired by http://www.pyimagesearch.com     
'''
def fourCornersSort(pts):           
    diff = np.diff(pts, axis=1)        
    summ = pts.sum(axis=1)                     
    return np.array([pts[np.argmin(summ)],                         
                      pts[np.argmax(diff)],                         
                      pts[np.argmax(summ)],                         
                      pts[np.argmin(diff)]])



def ExistDigit(Cell):
    '''
        Convert to Gray
    '''
    cell = cv2.cvtColor(Cell, cv2.COLOR_BGR2GRAY)  
    
    '''
        Blur the cell
    '''
    cell = cv2.GaussianBlur(cell,(5,5),0) 
    
    '''
        Binary the Image
    ''' 
    cell = cv2.adaptiveThreshold(cell,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_BINARY,11,5)

    '''
        Find edges of cell
    '''
    edges = cv2.Canny(cell, 75, 200)                
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:               
        (x, y, w, h) = cv2.boundingRect(cnt) 
        if w >= 10 and h >= 10:
            print("is Digit")
            cv2.drawContours(Cell, [cnt], -1, (255, 255, 0), 2)
            return True
    return False

'''
    Read Image from input
''' 
img = cv2.imread('image10.jpg')
image_copy = img
'''
    Convert the image to gray scale
'''
img = gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
plt.imshow(img,cmap='gray')
plt.show()

'''   
    Blur image
'''
img = cv2.GaussianBlur(img,(5,5),0)  

plt.imshow(img,cmap='gray')
plt.show()
'''
    Binary image
'''
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,3)
plt.imshow(img,cmap='gray')
plt.show()
'''
    Find Edges in Image
'''
edges = cv2.Canny(img, 75, 200)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    
plt.imshow(edges,cmap='gray')
plt.show()

'''
    Getting contours 
'''     
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

'''
    Finding contour of biggest rectangle   
    Otherwise return corners of original image
'''       
height = edges.shape[0]    
width = edges.shape[1]    
MAX_COUNTOUR_AREA = (width) * (height)

'''
    Page fill at least 0.1 of image, then saving max area found
'''    
maxAreaFound = MAX_COUNTOUR_AREA * 0.1

'''
    Saving page contour
'''    
pageContour = np.array([[0, 0], [0, height], [width, height], [width,0]])
   
for cnt in contours:               
    perimeter = cv2.arcLength(cnt, True)        
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    
    '''
        Page has 4 corners and it is convex        
        Page area must be bigger than maxAreaFound
    '''         
    if (len(approx) == 4 and cv2.isContourConvex(approx) 
        and maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):   
        maxAreaFound = cv2.contourArea(approx)            
        pageContour = approx
        cv2.drawContours(image_copy, [pageContour], -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.show()

'''
    Sort corners 
'''
sPoints = fourCornersSort(pageContour[:, 0]) 

'''
    Using Euclidean distance    
    Calculate maximum height and width  
'''  
height = max(np.linalg.norm(sPoints[0] - sPoints[1]),                 
             np.linalg.norm(sPoints[2] - sPoints[3]))
    
width = max(np.linalg.norm(sPoints[1] - sPoints[2]),                 
            np.linalg.norm(sPoints[3] - sPoints[0]))

'''
    Create target points
'''    
tPoints = np.array([[0, 0],                        
                    [0, height],                        
                    [width, height],                        
                    [width, 0]], np.float32)
'''
    getPerspectiveTransform() needs float32 
'''   
if sPoints.dtype != np.float32:        
    sPoints = sPoints.astype(np.float32)
    
'''
    Wraping perspective
'''    
M = cv2.getPerspectiveTransform(sPoints, tPoints)     
newImage = cv2.warpPerspective(image_copy, M, (int(width), int(height)))


'''
    Find width and height of Sudoku table
'''
width, height , channels = newImage.shape

'''
    Calculate width and height of each cells in Sudoku table 
'''
cell_width = int(width / 9)
cell_height = int(height / 9)

'''
    Calculate offset to get smaller the cell
'''
offset_width = int(cell_width * 0.15)
offset_height = int(cell_height * 0.15)

for i in range(0,9):
    for j in range(0,9):
        
        '''
            Find Cells
        '''
        cell= newImage[(i*cell_width)+offset_width:((i+1)*cell_width)-offset_width,(j*cell_height)+offset_height:((j+1)*cell_height)-offset_height]       
        b = ExistDigit(cell)
        '''
        display cell in Sudoku table
        '''   
        cv2.rectangle(newImage,(i*cell_height,j*cell_width),((i+1)*cell_height,(j+1)*cell_width),(0,255,0),3)

plt.imshow(newImage,cmap='gray')
plt.show()

'''
# Read data 
#train_images, train_labels

# find feature from train data
def feature_extractor(train_images):
    winSize = (32,32)
    blockSize = (12,12)
    blockStride = (5,5)
    cellSize = (12,12)
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
        im = cv2.resize(im, (32, 32), interpolation = cv2.INTER_AREA)
        h = np.zeros((225,1))
        h=hog.compute(im)
        h = h.flatten()
        descriptor.append(h)
        
    return np.float32(descriptor) 

#now train

def training(features, train_labels):
    
    #Set up SVM for OpenCV 3
    svm = cv2.ml.SVM_create()
    
    # Set SVM type
    svm.setType(cv2.ml.SVM_C_SVC)
    
    # Set SVM Kernel to Radial Basis Function (RBF) 
    svm.setKernel(cv2.ml.SVM_RBF)
    
    # Set parameter C
    svm.setC(2)
    
    # Set parameter Gamma
    svm.setGamma(1)
    
    # Train SVM on training data  
    svm.train(features, cv2.ml.ROW_SAMPLE, train_labels)
    
    return svm
#model = training(hog_featurs, train_labels)

def testing(model,test_data):
    testResponse = model.predict(test_data)[1].ravel()
    return testResponse

#prediction_label = testing(model, hog_featurs)
'''