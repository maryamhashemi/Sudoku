import cv2
from matplotlib import pyplot as plt
import numpy as np
from Sudoku.SVM import SVM
from Sudoku.KNN import KNN

class Image_Processing:
    
    SudokoTableImage = []
    
    def __init__(self, modelName, model):
        self.modelName = modelName
        self.model = model      
        
    def FourCornersSort(self,pts):
        '''
        Sort corners: top-left, bot-left, bot-right, top-right 
        Inspired by http://www.pyimagesearch.com     
        '''           
        diff = np.diff(pts, axis=1)        
        summ = pts.sum(axis=1)                     
        return np.array([pts[np.argmin(summ)], pts[np.argmax(diff)], pts[np.argmax(summ)], pts[np.argmin(diff)]])
        
    def ExistDigit(self,Cell):
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
                #cv2.drawContours(Cell, [cnt], -1, (255, 255, 0), 2)
                return True
        return False

    def FindSudokuTable(self,edges,image):  
        
        MaxRect = self.FindMaxRectContours(edges, image)       
        self.SudokoTableImage = self.CropSudokoTable(MaxRect, image)
        return self.SudokoTableImage
            
    def FindSudokuCells(self,image):
        
        '''
            Find width and height of Sudoku table
        '''
        width, height , channels = image.shape
        
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
        
        sudoku = np.zeros(shape = (9,9), dtype=np.int32)
        
        for i in range(0,9):
            for j in range(0,9):
                
                '''
                    Find Cells
                '''
                cell= image[(i*cell_width)+offset_width:((i+1)*cell_width)-offset_width,(j*cell_height)+offset_height:((j+1)*cell_height)-offset_height]       
                if(self.ExistDigit(cell)):
                    if(self.modelName == 'SVM'):
                        sudoku[i][j] = self.DigitRecognizerSVM(cell)
                    elif(self.modelName == 'KNN'):
                        sudoku[i][j] = self.DigitRecognizerKNN(cell)
                                
                '''
                display cell in Sudoku table
                '''   
                cv2.rectangle(image,(i*cell_height,j*cell_width),((i+1)*cell_height,(j+1)*cell_width),(0,255,0),3)
        
        self.Display(image) 
        return sudoku 
    
    def Display(self,img):
        plt.imshow(img,cmap='gray')
        plt.show()
        
    def FindEdges(self,img): 
        '''
            Convert the image to gray scale
        '''
        img = gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
        self.Display(img)
        '''   
            Blur image
        '''
        img = cv2.GaussianBlur(img,(5,5),0)  
        self.Display(img)
        '''
            Binary image
        '''
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,3)
        self.Display(img)
        '''
            Find Edges in Image
        '''
        edges = cv2.Canny(img, 75, 200)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
        self.Display(edges)
        
        return edges    
    
    def FindMaxRectContours(self,edges,image): 
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
                
        cv2.drawContours(image, [pageContour], -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        return pageContour
    
    def CropSudokoTable(self,MaxRect,image): 
        '''
            Sort corners 
        '''
        sPoints = self.FourCornersSort(MaxRect[:, 0]) 
        
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
        newImage = cv2.warpPerspective(image, M, (int(width), int(height)))
        return newImage
    
    def ExtractSudokuTable(self,filename):  
        '''
            Read Image from input
        ''' 
        img = cv2.imread(filename)
        edges = self.FindEdges(img)      
        SudokuTable = self.FindSudokuTable(edges, img)
        return self.FindSudokuCells(SudokuTable)
    
    def DigitRecognizerSVM(self,image):
        svm = SVM()
        image = svm.PreProcessingForSVM(image)
        image = np.reshape(image,(1,32,32))
        hog_feature = svm.feature_extractor(image)
        return svm.testing(self.model, hog_feature)
    
    def DigitRecognizerKNN(self,image):
        knn = KNN()
        image = knn.PreProcessingForKNN(image)
        image = np.reshape(image,(1,32,32))
        hog_feature = knn.feature_extractor(image)
        return knn.testing(self.model, hog_feature)

    def DisplaySolution(self,grid):
        '''
            Find width and height of Sudoku table
        '''
        width, height , channels = self.SudokoTableImage.shape
        
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
        
        for i,x in zip(range(0,9),'ABCDEFGHI'):
            for j,y in zip(range(0,9),'123456789'):
                          
                cell= self.SudokoTableImage[(i*cell_width)+offset_width:((i+1)*cell_width)-offset_width,(j*cell_height)+offset_height:((j+1)*cell_height)-offset_height]       
                if(self.ExistDigit(cell) == False):                    
                    '''
                    display digits in Sudoku table
                    '''   
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(self.SudokoTableImage,str(grid[x + str(y)]),((j*cell_height) + int(cell_height/2) - offset_height,(i*cell_width) + int(cell_width) - offset_width), font, 5,(255,255,0),5,cv2.LINE_AA)
        
        self.Display(self.SudokoTableImage) 