import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from ant import Ant


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


def ant_colony_opt(matrix, n_ants=10, max_iterations=23000):
    best_pheromone_matrix = np.ones(shape=(9, 9, 9)) / 9

    ants = [Ant(matrix.copy(), best_pheromone_matrix) for _ in range(n_ants)]
    print(ants[0].fitness())
    best_fitness = float('inf')
    best_ant = None

    history = []

    for _ in tqdm(range(max_iterations)):
        for i, ant in enumerate(ants):

            ant.solve_sudoku(epsilon_greedy=0.05)
            fitness = ant.fitness()

            if fitness < best_fitness:
                best_ant = copy.deepcopy(ant)
                best_fitness = fitness
                tqdm.write(str(best_fitness))

        for i in range(9):
            for j in range(9):
                best_pheromone_matrix[i][j][best_ant.current_solution[i, j] - 1] += 0.0005
                best_pheromone_matrix[i][j] /= np.sum(best_pheromone_matrix[i][j])

        history.append(best_fitness)
        if best_fitness == 0:
            # We found the solution, just stop
            break

    return best_ant.current_solution, history


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
    sudoku_name = "easiest"
    sudoku = np.loadtxt("sudokus/{}.txt".format(sudoku_name), dtype=np.int8)

    initial_board = copy.deepcopy(sudoku)

    n_ants = 10
    n_epochs = 3

    plt.figure()
    plt.title("sudoku {} with {} ants".format(sudoku_name, n_ants))
    plt.xlabel("Iteration")
    plt.ylabel("Mistakes")
    for e in range(n_epochs):
        best_solution, history = ant_colony_opt(initial_board, n_ants=n_ants)

        print('--------------')
        print('Initial Board')
        print_matrix(initial_board)

        print('--------------')
        print('Found Solution')
        print_matrix(best_solution)

        plt.plot(history, label="ACO_{}".format(e))

    plt.show()


if __name__ == "__main__":
    main()
