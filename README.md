# HMM

The code is written in Python 3. Before running the code, the packages in requirements.txt need to be installed:
    
    pip install -r requirements.txt
    
For training the HMM use the following command in terminal:

    python main.py train <train_dir>
    
where <train_dir> is the directory containing all training files (./train)

For testing the HMM use the following command in terminal:

    python main.py test <test_dir> <test_file>
    
where <test_dir> is the directory containing the test file (./test)
and <test_file> is the name of the test file in the given directory
