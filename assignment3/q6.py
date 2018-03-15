# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:08:54 2018

@author: Pauline LAURON
"""

import numpy as np
import random
import copy
from tqdm import tqdm


class Ant(object):

    def __init__(self, unsolved_sudoku, pheromone_matrix):
        self.unsolved_sudoku = unsolved_sudoku.copy()
        self.pheromone_matrix = pheromone_matrix
        self.current_solution = self.unsolved_sudoku.copy()

    def solve_sudoku(self):
        # reset the sudoku
        self.current_solution = self.unsolved_sudoku.copy()

        indexes = np.asarray([coord for sublist in [[(i,j) for j in range(9)] for i in range(9)] for coord in sublist])
        np.random.shuffle(indexes)

        possible = np.ones((9,9), dtype=bool)
        for i, row in enumerate(possible):
            for j in range(9):
                if self.unsolved_sudoku[i][j] != 0:
                    # make sure that we only pick new numbers for each row
                    possible[i][self.unsolved_sudoku[i][j] - 1] = 0

        for index in indexes:
            if self.current_solution[index[0]][index[1]] == 0:
                pheromones = self.pheromone_matrix[index[0]][index[1]]
                probs = pheromones* possible[index[0]]
                probs = probs / np.sum(probs)
                number = np.random.choice(range(1,10), p=probs)
                possible[index[0]][number - 1] = 0
                self.current_solution[index[0]][index[1]] = number



    def fitness(self):
        return self.row_violations() + self.column_violations() + self.subgrid_violations()

    def row_violations(self):
        number_updates = 0
        total_violations = 0
        for row in self.current_solution:
            occurences = np.bincount(row)
            total_violations += np.sum(occurences[occurences != 1])


        return total_violations


    def column_violations(self):
        m = self.current_solution.copy()
        m = m.T
        number_updates = 0
        total_violations = 0
        for row in m:
            occurences = np.bincount(row)
            total_violations += np.sum(occurences[occurences != 1])

        return total_violations

    def extract_subgrid(self, i, j):
        i *= 3
        j *= 3

        return np.reshape(self.current_solution[i:i+3, j:j+3], 9)


    def subgrid_violations(self):
        total_violations = 0
        for i in range(3):
            for j in range(3):
                row = self.extract_subgrid(i,j)
                occurences = np.bincount(row)
                total_violations += np.sum(occurences[occurences != 1])

        return total_violations


def isRowValid(board):
    for i in range(len(board)):
        row = []
        for j in range(len(board[0])):
            row.append(board[i][j].number)
        for k in range(1, 10):
            if row.count(k) > 1:
                return False

    # check if every cell has at least 1 candidate
    for i in range(len(board)):
        for j in range(len(board[0])):
            if sum(board[i][j].candidate) == 0:
                return False

    return True


def isColumnValid(board):
    for j in range(len(board[0])):
        column = []
        for i in range(len(board[0])):
            column.append(board[i][j].number)

        for i in range(1, 10):
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

        for j in range(1, 10):
            if list_value.count(j) > 1:
                return False
    return True


def valid(board):
    if isRowValid(board) and isColumnValid(board) and isSubGridValid(board):
        return True
    else:
        return False


def ant_colony_opt(matrix, n_ants = 20, max_iterations = 23000):
    best_pheromone_matrix = np.ones(shape=(9,9,9)) / 9

    ants = [Ant(matrix.copy(), best_pheromone_matrix)]
    print(ants[0].fitness())
    best_fitness = 90000
    best_ant = None

    for i in tqdm(range(max_iterations)):
        for i, ant in enumerate(ants):

            ant.solve_sudoku()
            fitness = ant.fitness()

            if fitness < best_fitness:
                best_ant = copy.deepcopy(ant)
                best_fitness = fitness
                tqdm.write(str(best_fitness))

        for i in range(9):
            for j in range(9):
                best_pheromone_matrix[i][j][best_ant.current_solution[i,j] - 1] += 0.00005
                best_pheromone_matrix[i][j] /= np.sum(best_pheromone_matrix[i][j])



    print(best_fitness)

    print_matrix(best_ant.current_solution)






def print_matrix(matrix):
    for row in range(len(matrix)):
        print(matrix[row][0], matrix[row][1], matrix[row][2], matrix[row][3],
              matrix[row][4], matrix[row][5], matrix[row][6], matrix[row][7],
              matrix[row][8])



def sudoku_completed(board):
    for row in range(len(board)):
        for column in range(len(board[0])):
            if board[row][column].number == 0:
                return False

    return True


def main():
    sudoku = np.loadtxt("s01b.txt", dtype=np.int8)
    matrix = sudoku
    print_matrix(matrix)

    initialBoard = copy.deepcopy(matrix)
    current_it = 0

    ant_colony_opt(matrix)

        # print('--------------')
        # print_matrix(matrix)

    print('--------------')
    print('FINALE')
    print_matrix(matrix)


if __name__ == "__main__":
    main()
