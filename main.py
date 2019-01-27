from Sudoku.Image_Processing import Image_Processing
from Sudoku.SVM import SVM
from Sudoku.Sudoko_Solver import Sudoko_Solver
from Sudoku.KNN import KNN
from Sudoku.BruteForce import BruteForce
import glob
import numpy as np
from datetime import datetime

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
KNNmodel = knn.GetTrainedModel()

svm = SVM()
SVMmodel = svm.GetTrainedModel()


File_Path = './SudokuDateset/*.jpg'
files = glob.glob(File_Path)
TableNumber = len(files)
SolveNumber = 0

start_time = datetime.now()
for filename in files:
    print(filename)
    imgProc = Image_Processing('KNN', KNNmodel)
    grid = imgProc.ExtractSudokuTable(filename)
    
    test_lables = ReadDatFile(filename)
    CalcConfusionMatrix(test_labels= test_lables, prediction_label= grid)
    
    solver = Sudoko_Solver()
    grid = solver.solve(grid)
    
    if(grid != False): 
        imgProc.DisplaySolution(grid)
        SolveNumber +=1
'''
start_time = datetime.now()
for filename in files:
    print(filename)
    imgProc = Image_Processing('SVM', SVMmodel)
    grid = imgProc.ExtractSudokuTable(filename)
    
    test_lables = ReadDatFile(filename)
    CalcConfusionMatrix(test_labels= test_lables, prediction_label= grid)
    
    if(filename != './SudokuDateset\Sudoku (87).jpg'):
        solver = BruteForce(grid)
        solution = solver.solve()
        
        if(solution != False): 
            #imgProc.DisplaySolution(grid)
            SolveNumber +=1
'''  
              
print('Algorithm Time = ',  datetime.now() - start_time)        
print('Confusion Matrix = ')
print(confusion_matrix)
print()

print('Mean Accuracy = ')
print(CalcMeanAccuracy())
print()

print('Total Table = ')
print(TableNumber)
print()

print('Solved Table = ')
print(SolveNumber)
print()

print('Efficiency = ')
print(SolveNumber / TableNumber) 
print() 
