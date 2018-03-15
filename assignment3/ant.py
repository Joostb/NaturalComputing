import numpy as np

class Ant(object):

    def __init__(self, unsolved_sudoku, pheromone_matrix):
        self.unsolved_sudoku = unsolved_sudoku.copy()
        self.pheromone_matrix = pheromone_matrix
        self.current_solution = self.unsolved_sudoku.copy()

    def solve_sudoku(self, epsilon_greedy=0.2):
        # reset the sudoku
        self.current_solution = self.unsolved_sudoku.copy()

        indexes = np.asarray([coord for sublist in [[(i, j) for j in range(9)] for i in range(9)] for coord in sublist])
        np.random.shuffle(indexes)

        possible = np.ones((9, 9), dtype=bool)
        for i, row in enumerate(possible):
            for j in range(9):
                if self.unsolved_sudoku[i][j] != 0:
                    # make sure that we only pick new numbers for each row
                    possible[i][self.unsolved_sudoku[i][j] - 1] = 0

        for index in indexes:
            if self.current_solution[index[0]][index[1]] == 0:
                pheromones = self.pheromone_matrix[index[0]][index[1]]
                probs = pheromones
                if np.random.rand() < epsilon_greedy:
                    probs = np.ones(9) / 9

                probs = probs * possible[index[0]]
                probs = probs / np.sum(probs)
                number = np.random.choice(range(1, 10), p=probs)
                possible[index[0]][number - 1] = 0
                self.current_solution[index[0]][index[1]] = number

    def fitness(self):
        return self.column_violations() + self.subgrid_violations()

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
        total_violations = 0
        for row in m:
            occurences = np.bincount(row)
            total_violations += np.sum(occurences[occurences != 1])

        return total_violations

    def extract_subgrid(self, i, j):
        i *= 3
        j *= 3

        return np.reshape(self.current_solution[i:i + 3, j:j + 3], 9)

    def subgrid_violations(self):
        total_violations = 0
        for i in range(3):
            for j in range(3):
                row = self.extract_subgrid(i, j)
                occurences = np.bincount(row)
                total_violations += np.sum(occurences[occurences != 1])

        return total_violations
