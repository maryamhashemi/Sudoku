from Sudoku.Image_Processing import Image_Processing
from Sudoku.SVM import SVM
from Sudoku.Sudoko_Solver import Sudoko_Solver


svm = SVM()
model = svm.GetTrainedModel()

filename = './SudokuDateset/sample.jpg'
imgProc = Image_Processing(model)
print(imgProc.ExtractSudokuTable(filename))
