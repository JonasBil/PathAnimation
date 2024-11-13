import pygame
import math
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

Red = (255, 0, 0)
Green = (0, 255, 0)
Blue = (0, 0, 255)
Yellow = (255, 255, 0)
White = (255, 255, 255)
Black = (0, 0, 0)
Purple = (128, 0, 128)
Orange = (255, 165, 0)
Grey = (128, 128, 128)
Turquoise = (64, 224, 208)

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = White
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == Red

    def is_open(self):
        return self.color == Green

    def is_barrier(self):
        return self.color == Black

    def is_start(self):
        return self.color == Orange

    def is_end(self):
        return self.color == Turquoise

    def reset(self):
        self.color = White

    def make_closed(self):
        self.color = Red

    def make_open(self):
        self.color = Green

    def make_barrier(self):
        self.color = Black

    def make_start(self):
        self.color = Orange

    def make_end(self):
        self.color = Turquoise

    def make_path(self):
        self.color = Purple

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid): # Updating the neighbors of the spot.
        self.neighbors = [] # Resetting the neighbors.
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # If the spot below the current spot is not a barrier. DOWN
            self.neighbors.append(grid[self.row + 1][self.col]) # Add the spot to the neighbors list.

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    # Less than operator, to compare two Spot objects with each other.
    def __lt__(self, other):
        return False


def h(p1, p2):
    # Using Manhattan distance to calculate the heuristic value.
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw): # To reconstruct the path.
    while current in came_from: # While we have not reached the start node.
        current = came_from[current] # Get the node from where we came to reach the current node.
        current.make_path()  # Make the node part of the path.
        draw() # Draw the grid.


# The main algorithm function.
def algorithm(draw, grid, start, end):
    count = 0 # To keep track of the order of the nodes in the priority queue.
    open_set = PriorityQueue() # Priority queue to store the nodes.
    open_set.put((0, count, start)) # Putting the start node in the priority queue. 0 is the f-score, count is the order when we inputted the value in the cue, and start is the node.
    came_from = {} # To keep track of the path, where we came from to reach a node.
    g_score = {spot: float("inf") for row in grid for spot in row} # To keep track of the g-score of each node. Initially set to infinity.
    g_score[start] = 0 # The g-score of the start node is 0.
    f_score = {spot: float("inf") for row in grid for spot in row} # To keep track of the f-score of each node. Initially set to infinity.
    f_score[start] = h(start.get_pos(), end.get_pos()) # The f-score of the start node is the heuristic value of the start node.

    open_set_hash = {start} # To keep track of the nodes in the priority queue, as we cannot check if a node is in the priority queue or not.

    while not open_set.empty(): # While the priority queue is not empty. When the priority queue is empty, it means that we have checked all the nodes.
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Because this is a while loop, we need a way to stop the loop when something goes wrong.
                pygame.quit()

        current = open_set.get()[2] # Get the node with the lowest f-score.
        open_set_hash.remove(current) # Remove the node from the priority queue.

        if current == end: # If the current node is the end node, we have reached the end node.
            reconstruct_path(came_from, end, draw) # Reconstruct the path.
            end.make_end() # Make the end node the end node otherwise the end node will be part of the path and will be colored purple.
            return True

        for neighbor in current.neighbors: # For every neighbor of the current node.
            temp_g_score = g_score[current] + 1 # The g-score of the neighbor is the g-score of the current node + 1, we assume that the distance between two nodes is 1.

            if temp_g_score < g_score[neighbor]: # If the new g-score is less than the previous g-score, we have found a better path.
                came_from[neighbor] = current # Update the path.
                g_score[neighbor] = temp_g_score # Update the g-score.
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos()) # Update the f-score, the f-score is the g-score + the heuristic value.
                if neighbor not in open_set_hash: # If the neighbor is not in the priority queue.
                    count += 1 # Increment the count.
                    open_set.put((f_score[neighbor], count, neighbor)) # Put the neighbor in the priority queue.
                    open_set_hash.add(neighbor) # Add the neighbor to the set, we don't care about the value, we just want to know if the neighbor is in the set or not.
                    neighbor.make_open() # Make the neighbor open, to show that we are considering the neighbor.

        draw() # Draw the grid.

        if current != start: # If the current node is not the start node.
            current.make_closed() # Make the current node closed, to show that we have considered the node.

    return False # If the priority queue is empty and we have not reached the end node, then there is no path.

# Making the grid.
def make_grid(rows, width):
    grid = []
    gap = width // rows # Width of each cell, calculated by dividing the total width by the number of rows.
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows) #i is the row, j is the column, gap is the width of each cell, rows is the total number of rows.
            grid[i].append(spot) # Appending the spot object to the grid.
    return grid

# Drawing the grid lines. We now just have cubes with no lines separating them.
def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows): # For every row, draw a horizontal line.
        pygame.draw.line(win, Grey, (0, i * gap), (width, i * gap))
        for j in range(rows): # For every column, draw a vertical line.
            pygame.draw.line(win, Grey, (j * gap, 0), (j * gap, width))

# Drawing the grid.
def draw(win, grid, rows, width):
    win.fill(White) # Filling the window with white color, to clear the previous state of the grid.

    for row in grid:
        for spot in row:
            spot.draw(win) # Drawing each spot in the grid, see Spot class.

    draw_grid(win, rows, width) # Drawing the grid lines.
    pygame.display.update() # Updating the display.

# Getting the clicked position of the mouse on the grid, and returning the row and column of the spot.
def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap # Getting the row by dividing the y-coordinate by the width of each cell.
    col = x // gap # Getting the column by dividing the x-coordinate by the width of each cell.

    return row, col

# Main function to run the A* algorithm. Here we will implement the A* algorithm and setting up the events.
def main(win, width):
    ROWS = 50 # Number of rows in the grid
    grid = make_grid(ROWS, width) # Making the grid, defined above.

    start = None # keep track of the start node
    end = None # keep track of the end node

    run = True # To keep the game running

    while run:
        draw(win, grid, ROWS, width) # Drawing the grid.
        for event in pygame.event.get(): # Getting the events. Events are anything that happens in the game, like mouse clicks, key presses, etc.
            if event.type == pygame.QUIT: # If the user clicks the close button on the window, the game will stop running.
                run = False # Stop the game.

            if pygame.mouse.get_pressed()[0]: # If the left mouse button is pressed.
                pos = pygame.mouse.get_pos() # Get the position of the mouse.
                row, col = get_clicked_pos(pos, ROWS, width) # Get the row and column of the spot where the mouse was clicked, see get_clicked_pos function.
                spot = grid[row][col]
                if not start and spot != end: # If the start node is not set.
                    start = spot
                    start.make_start() # Make the spot the start node.

                elif not end and spot != start: # If the end node is not set.
                    end = spot
                    end.make_end() # Make the spot the end node.

                elif spot != end and spot != start: # If the spot is not the start or end node.
                    spot.make_barrier() # Make the spot a barrier.

            elif pygame.mouse.get_pressed()[2]: # If the right mouse button is pressed.
                pos = pygame.mouse.get_pos() # Get the position of the mouse.
                row, col = get_clicked_pos(pos, ROWS, width) # Get the row and column of the spot where the mouse was clicked, see get_clicked_pos function.
                spot = grid[row][col] # Get the spot object at the row and column.
                spot.reset() # Reset the spot.
                if spot == start: # If the spot is the start node.
                    start = None # Reset the start node.
                elif spot == end: # If the spot is the end node.
                    end = None # Reset the end node.

            if event.type == pygame.KEYDOWN: # If a key is pressed.
                if event.key == pygame.K_SPACE and start and end: # If the space key is pressed and the start and end nodes are set.
                    for row in grid: # For every row in the grid.
                        for spot in row: # For every spot in the row.
                            spot.update_neighbors(grid) # Update the neighbors of the spot.

                    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end) # Run the algorithm function, see below.

                if event.key == pygame.K_c: # If the c key is pressed, clear the grid.
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)


    pygame.quit() # Quit the game.


main(WIN, WIDTH) # Run the main function.


