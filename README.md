# Perceptron

Binary Perceptron neural network that imports data to train and classify 3 classes of data and classify
Test data.
Separate perceptron presents the resulting weights and accuracy against the training and test datasets.
Several tests are conducted:
* Individual comparisons (class 1 vs class 2 etc.)
* one vs. rest (class 1 vs 2/3 etc.)
* one vs. rest with an added L2 regularisation with coefficients between 0.01-100
___
Created and tested in PyCharm Community Edition 4.5.2

Ensure the following software applications is available:
    Python 3.6.1
    Numpy 1.13.1
    (Optional) Python IDE, i.e. PyCharm, IDLE, Visual studio etc

Required files:
    Perceptron.py
    test.data
    train.data

Please save perceptron file 'Perceptron.py' and required test.data and train.data files in the same selected folder.
If path or data file name requires adjustment, please adjust variables 'trainFile' and 'testFile' respectively within the 'Perceptron.py' script.

Depending on your running tool, please follow methods accordingly:

Command Prompt (cmd.exe):
1) Open command prompt, and change directory to the saved file path of the python script using 'cd'. 
e.g.    M:>cd Perceptron
    M:\Perceptron>

2) Input 'python Perceptron.py' in the window to run the script.
e.g.    M:\Perceptron>python Perceptron.py


IDE (PyCharm, IDLE etc.):
1) In selected IDE, open 'Perceptron.py' file in the saved directory.

2) Run code from IDE, depending on the program:
e.g.    IDLE - Run > Run Module (F5)
    PyCharm - Run > Run... (Alt+Shift+F10)
                Select 'Perceptron'
