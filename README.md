# Sudoku_Solver_tf

# Introduction:
1. Sudoku is one of the most popular logic-based number-placement puzzle game. The literal meaning of "Su-doku" in Japanese is "the number that is single".

2. The objective is to fill a nine-by-nine (9x9) grid with digits so that each row, column and 3x3 section contain number between 1 and 9, with each number used once and only once in each section. The Sudoku game players are provided with partially filled grid meant to be solved.

3. To solve sudoku one doesn't require the knowledge of mathematics but require the logic and reasoning. Solving Sudoku Puzzles daily helps with your brain. It improves the concentration and logical thinking. One can look for sudoku puzzles given in Newspapers or can play them online provided by many websites.

# About:
This program serves as a way to calculate the solution to any 9x9 sudoku puzzle image. It identifies the puzzle through the image, processes it uses OpenCV, runs against a neural network to predict the digits, and runs an efficient sudoku solver to determine the answer. It then displays the answer on the in the terminal if it is solvable and also save the solved image in results folder.

# Relevant Packages:
opencv-python: 4.3.0.36
numpy: 1.19.1
tensorflow: 2.2.0
sklearn: 0.0
keras: 2.3.1

# Execution:
1. clone the repository from this link:        
        
        https://github.com/rajban94/Sudoku_Solver_tf.git

2. open the terminal and write the following command:           
        
        python3 main.py --image 'image path' --output_dir results/
