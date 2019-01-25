from Sudoku.Image_Processing import Image_Processing
from Sudoku.SVM import SVM
from Sudoku.Sudoko_Solver import Sudoko_Solver
from Sudoku.KNN import KNN


'''knn = KNN()
model = knn.GetTrainedModel()

filename = './SudokuDateset/sample.jpg'
imgProc = Image_Processing('KNN' , model)
print(imgProc.ExtractSudokuTable(filename))
'''
svm = SVM()
model = svm.GetTrainedModel()

filename = './SudokuDateset/sample.jpg'
imgProc = Image_Processing('SVM', model)
grid = imgProc.ExtractSudokuTable(filename)


solver = Sudoko_Solver()
grid = solver.solve(grid)

imgProc.DisplaySolution(grid)