# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:08:54 2018

@author: Pauline LAURON
"""

import numpy as np
import random
import copy

class Value(object):
    
    def __init__(self, number, row, column):
        self.number = int(number)
        self.row = row
        self.column = column
        self.candidate = [0,0,0,0,0,0,0,0,0]
        self.pheromone = [1,1,1,1,1,1,1,1,1]
        
        if(self.number > 0):
            self.candidate[self.number-1] = 1
        
        if self.number == 0 :
            self.candidate = [1,1,1,1,1,1,1,1,1]
            
    def setCandidate(self, index, value):
        
        self.candidate[index] = value
        
        if sum(self.candidate) == 1:
            for i, value in enumerate(self.candidate):
                if value == 1:
                    self.number = i+1

def row_simplification(matrix):
    number_updates = 0
    
    for row in range(len(matrix)):
        numbers_in_row = []
        for column in range(len(matrix[0])):
            value = matrix[row][column]
            if value.number > 0 :
                numbers_in_row.append(value.number)
        for column in range(len(matrix[0])):
            value = matrix[row][column]
            if value.number == 0 :
                for number in numbers_in_row: 
                    if value.candidate[number-1] != 0 :
                        value.setCandidate(number-1, 0) #value.candidate[number-1] = 0
                        number_updates = number_updates + 1                 
    
    return number_updates

def column_simplification(matrix):
    number_updates = 0
    
    for column in range(len(matrix[0])):
        numbers_in_column = []
        for row in range(len(matrix)):
            value = matrix[row][column]
            if value.number > 0 :
                numbers_in_column.append(value.number)
        for row in range(len(matrix)):
            value = matrix[row][column]
            if value.number == 0 :
                for number in numbers_in_column: 
                    if value.candidate[number-1] != 0 :
                        value.setCandidate(number-1, 0) #value.candidate[number-1] = 0
                        number_updates = number_updates + 1                 
    
    return number_updates

def extract_subgrid(i, matrix):
            
    row_count = int(i/3)
    column_count = i%3 * 3
    
    subgrid = [[] for _ in range(3)]
    
    for row in range(3):
        for column in range(3):
            subgrid[row].append(matrix[row_count*3+row][column_count+column])
        
    return subgrid
    
def subgrid_simplification(matrix):
     number_updates = 0
    
     for i in range(0,9):
        subgrid = extract_subgrid(i, matrix)
        numbers_in_subgrid = []
        for row in range(len(subgrid)):
            for column in range(len(subgrid[0])):
                value = subgrid[row][column]
                if value.number > 0 :
                    numbers_in_subgrid.append(value.number)
        for row in range(len(subgrid)):
            for column in range(len(subgrid[0])):
                value = subgrid[row][column]
                if value.number == 0 :
                    for number in numbers_in_subgrid: 
                        if value.candidate[number-1] != 0 :
                            value.setCandidate(number-1, 0) #value.candidate[number-1] = 0
                            number_updates = number_updates + 1                 
    
     return number_updates
 
def isRowValid(board): 
    
    for i in range(len(board)): 
        row = []
        for j in range(len(board[0])):
            row.append(board[i][j].number)
        for k in range(1,10):
            if row.count(k) > 1:
                return False

    #check if every cell has at least 1 candidate
    for i in range(len(board)):
        for j in range(len(board[0])):
            if sum(matrix[i][j].candidate) == 0 : 
                return False
            
    return True
    

def isColumnValid(board):
    
    for j in range(len(board[0])):
        column = []
        for i in range(len(board[0])):
            column.append(board[i][j].number)

        for i in range(1,10):
            if column.count(i) > 1:
                return False
            
    return True

def isSubGridValid(board):
    
    for i in range(len(board)):
        subgrid = extract_subgrid(i, board)
        
        list_value = []
        for row in range(len(subgrid)):
            for column in range(len(subgrid[0])):
                list_value.append(subgrid[row][column].number)
            
        for j in range(1,10):
            if list_value.count(j) > 1:
                return False
    return True
    
 
def valid(board):

    if isRowValid(board) and isColumnValid(board) and isSubGridValid(board): 
        return True 
    else: 
        return False
 
    
def ant_colony_opt(matrix, initialBoard, current_it):
    award = 1
    punishment = 0.005
    unassignedSet= []
    initialBoardLocal = copy.deepcopy(matrix)
    
    #list of unassigned cell
    for row in range(len(initialBoard)):
        for column in range(len(initialBoard[0])):
            value = matrix[row][column]
            if (value.number == 0):
                unassignedSet.append(value)
                
    if len(unassignedSet) == 0:
        current_it = current_it+1
        return 0
                
    #randomly select an unassigned cell
    selectedUCell = unassignedSet[random.randint(0, len(unassignedSet)-1)]
    candidates = selectedUCell.candidate
    maxPheromone = -float('inf')
    aNumber = 0
    
    #assign cell to the candidate with the highest pheronome accumulation
    for k in range(len(initialBoard)):
        if(candidates[k] == 1):
            pheromone = selectedUCell.pheromone[k]
            if pheromone > maxPheromone:
                maxPheromone = pheromone
                aNumber = k+1  
                
    #test if its compatible
    if aNumber > 0:
        selectedUCell.number = aNumber          
    else: 
        if sum(selectedUCell.candidate) == 0:
            current_it = current_it+1
            return current_it
        for i in selectedUCell.candidate:
            if(i == 1):    
                selectedUCell.number = i+1
    
    if valid(matrix) :   
        simplify(matrix)
        
        if valid(matrix):
            selectedUCell.pheromone[aNumber-1] = selectedUCell.pheromone[aNumber-1] + award
        else:
            for row in range(len(matrix)):
                for column in range(len(matrix[0])):
                    matrix[row][column].number = initialBoardLocal[row][column].number
                    matrix[row][column].candidate = initialBoardLocal[row][column].candidate
                    
            current_it = current_it+1
    else:
        selectedUCell.pheromone[aNumber-1] = max(selectedUCell.pheromone[aNumber-1] - punishment, 0)
        for row in range(len(matrix)):
            for column in range(len(matrix[0])):
                matrix[row][column].number = initialBoardLocal[row][column].number
                matrix[row][column].candidate = initialBoardLocal[row][column].candidate
                
        current_it = current_it+1
    
    if(current_it > 1000):
        #no possible solution, try from the beginning
        for row in range(len(matrix)):
            for column in range(len(matrix[0])):
                matrix[row][column].number = initialBoard[row][column].number
                matrix[row][column].candidate = initialBoard[row][column].candidate
        current_it = 0
                
    return current_it
    
def print_matrix(matrix):
    
    for row in range(len(matrix)):
            print(matrix[row][0].number, matrix[row][1].number, matrix[row][2].number, matrix[row][3].number, matrix[row][4].number, matrix[row][5].number, matrix[row][6].number, matrix[row][7].number, matrix[row][8].number)

def simplify(matrix):
    while True :
        nb_updates = 0
        nb_updates = nb_updates + row_simplification(matrix)
        nb_updates = nb_updates + column_simplification(matrix)
        nb_updates = nb_updates + subgrid_simplification(matrix)
    
        if (nb_updates == 0):
            break
        
def sudoku_completed(board): 
    
    for row in range(len(board)): 
        for column in range(len(board[0])):
            if (board[row][column].number == 0):
                return False
            
    return True
        
        

sudoku = np.loadtxt("s11a.txt")

matrix = [[] for _ in range(len(sudoku[0]))]
for row in range(len(sudoku)):
    for column in range(len(sudoku[0])):
        matrix[row].append(Value(sudoku[row][column], row, column))

print_matrix(matrix)

initialBoard = copy.deepcopy(matrix)
current_it = 0

while sudoku_completed(matrix) == False or sudoku_completed(matrix) == False:
    simplify(matrix)
    current_it = ant_colony_opt(matrix, initialBoard, current_it)
    
    print('--------------')    
    print_matrix(matrix)
    
    
print('--------------')    
print('FINALE') 
print_matrix(matrix)

