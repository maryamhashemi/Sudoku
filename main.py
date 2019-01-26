from Sudoku.Image_Processing import Image_Processing
from Sudoku.SVM import SVM
from Sudoku.Sudoko_Solver import Sudoko_Solver
from Sudoku.KNN import KNN
import glob
import numpy as np

confusion_matrix = np.zeros((10,10),dtype = 'int')

def CalcConfusionMatrix(test_labels,prediction_label): 
    
    test_labels = test_labels.flatten()
    prediction_label = prediction_label.flatten()
              
    for i in range(0,len(prediction_label)): 
        for j in range(0,10):      
            if (prediction_label[i] == j):
                confusion_matrix[test_labels[i]][j] += 1
                
def CalcMeanAccuracy(): 
      
    mean_accuracy =0.0       
    for i in range(0,10):
        Sum = 0
        for j in range(0,10):
            Sum = Sum + confusion_matrix[i][j]                      
        accuracy = confusion_matrix[i][i] / Sum
        mean_accuracy = mean_accuracy + accuracy
    
    mean_accuracy = mean_accuracy / 10
    return mean_accuracy
        
def ReadDatFile(filename):
    filename = filename.replace('jpg', 'dat')
    data = np.loadtxt( filename )
    data = data.astype(int)
    return data
    
    
knn = KNN()
model = knn.GetTrainedModel()

#svm = SVM()
#model = svm.GetTrainedModel()


File_Path = './SudokuDateset/*.jpg'
files = glob.glob(File_Path)

for filename in files:
    
    imgProc = Image_Processing('KNN', model)
    grid = imgProc.ExtractSudokuTable(filename)
    
    print(grid)
    test_lables = ReadDatFile(filename)
    CalcConfusionMatrix(test_labels= test_lables, prediction_label= grid)
    
    solver = Sudoko_Solver()
    grid = solver.solve(grid)
    
    if(grid != False): 
        imgProc.DisplaySolution(grid)

print(confusion_matrix)
print(CalcMeanAccuracy()) 
 