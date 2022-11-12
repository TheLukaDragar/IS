
from concurrent.futures.thread import *
from threading import Lock
from random import randint
import pygame
from concurrent.futures import as_completed

import Solve as Solve

#import ascompleted


lock = Lock()

maze_dict = {
    "#": 0,
    ".": 1,
    "S": 2,
    "E": 3,
    "T": 4,
}


def make_move(move_direction, current_pos,maze):
    #we move in the maze and check if move is valid
    #if move is valid we return new position and valid flag
    #if move is not valid we return current position and invalid flag
    new_pos = current_pos

    if move_direction == 0:
        #up
        new_pos = (current_pos[0] - 1, current_pos[1])
    elif move_direction == 1:
        #down
        new_pos = (current_pos[0] + 1, current_pos[1])

    elif move_direction == 2:
        #left
        new_pos = (current_pos[0], current_pos[1] - 1)

    elif move_direction == 3:
        #right
        new_pos = (current_pos[0], current_pos[1] + 1)

    #check if move is valid
    if new_pos[0] < 0 or new_pos[0] >= maze.shape[0] or new_pos[1] < 0 or new_pos[1] >= maze.shape[1]:
        #move is not valid because we are out of maze
        return current_pos, False

    if maze[new_pos[0], new_pos[1]] == maze_dict["#"]:
        #move is not valid because we hit a wall
        return current_pos, False

    #move is valid
    return new_pos, True

def simulate_chromosomes(maze,maze_start,maze_end,chromosomes, title="",auto=True,draw=False,custom_colors=[],crossover_data=None):
    #use pygame to simulate chromosome is a numpy array of moves
    
    #open pygame window
    pygame.init()
    screen = pygame.display.set_mode((maze.shape[1] * 20, maze.shape[0] * 20))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    
    game_finished = False
    running=True
    step=0
    current_positions = [maze_start for i in range(len(chromosomes))]
    colors= custom_colors if custom_colors else [(randint(0, 255), randint(0, 255), randint(0, 255)) for i in range(len(chromosomes))]
    
    #aray of arrays of visited points
    visited = [[maze_start] for i in range(len(chromosomes))]

    if crossover_data:
        #we have crossover data so we draw the crossover point
       intersection_point=crossover_data["intersection_point"]
       intersection_point_chromosome1_index=crossover_data["c1_rand_intersection_index"]
       intersection_point_chromosome2_index=crossover_data["c2_rand_intersection_index"]

    #must be different colors for each chromosome\
    while running:

        #draw maze
        for i in range(maze.shape[0]):

            for j in range(maze.shape[1]):
                if maze[i, j] == maze_dict["#"]:
                    pygame.draw.rect(screen, (0, 0, 0), (j * 20, i * 20, 20, 20))
                elif maze[i, j] == maze_dict["S"]:
                    pygame.draw.rect(screen, (0, 255, 0), (j * 20, i * 20, 20, 20))
                elif maze[i, j] == maze_dict["E"]:
                    pygame.draw.rect(screen, (255, 0, 0), (j * 20, i * 20, 20, 20))
                elif maze[i, j] == maze_dict["T"]:
                    pygame.draw.rect(screen, (255, 255, 0), (j * 20, i * 20, 20, 20))
                else:
                    pygame.draw.rect(screen, (255, 255, 255), (j * 20, i * 20, 20, 20))


        size = 20

        if draw:
            for i in range(len(visited)):
                for j in range(len(visited[i])):
                    
                
                    #make color more transparent
                    colorr = (colors[i][0], colors[i][1], colors[i][2], 100)
                    pygame.draw.rect(screen, colorr, (visited[i][j][1] * 20 + (20 - size/2) // 2, visited[i][j][0] * 20 + (20 - size/2) // 2, size/2, size/2))


        #draw current positions with different colors
      
        for i in range(len(current_positions)):
            #make each one smaller so we can see them
            size = 20 - 2 * i
            pygame.draw.rect(screen, colors[i], (current_positions[i][1] * 20 + (20 - size) // 2, current_positions[i][0] * 20 + (20 - size) // 2, size, size))
            

        #draw crossover data
        if crossover_data:
            if step >= intersection_point_chromosome1_index or step >= intersection_point_chromosome2_index:
            #draw intersection point yellow
                pygame.draw.rect(screen, (255, 255, 0), (intersection_point[1] * 20 + (20 - size) // 2, intersection_point[0] * 20 + (20 - size) // 2, size, size))
            if auto:
                #stop auto when intersection point is reached
                if step == intersection_point_chromosome1_index or step == intersection_point_chromosome2_index:
                    auto=False

                    #check if point is in both visited arrays
                    if intersection_point in visited[0] and intersection_point in visited[1]:
                        print("Intersection point is in both visited arrays")
                    else:
                        print("Intersection point is not in both visited arrays")
        
                    

            

                
        #wait for user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return
               
            if event.type == pygame.KEYDOWN:

                if game_finished:
                    game_finished = False

                if event.key == pygame.K_SPACE:
                    auto = not auto

                

                #if auto is true we will simulate chromosome automatically
                #if auto is false we will simulate chromosome step by step
                #of w is pressed we will make a step
                if event.key == pygame.K_w:
                    auto = False
                    
                    for i in range(len(current_positions)):
                        current_positions[i], _ = make_move(chromosomes[i][step], current_positions[i], maze)
                    step += 1

                #q is pressed we will quit
                if event.key == pygame.K_q:
                    running = False
                    pygame.quit()
                    return

                #reset
                if event.key == pygame.K_r:
                    current_positions = [maze_start for i in range(len(chromosomes))]
                    step = 0
                    visited = [[maze_start] for i in range(len(chromosomes))]


                

        if auto and not game_finished:
            #make move
            for i in range(len(current_positions)):
                current_positions[i], _ = make_move(chromosomes[i][step], current_positions[i], maze)
            step += 1

            #delay
            pygame.time.delay(50)


        if step >= len(chromosomes[0]):
            #we reached the end of chromosome
            game_finished=True
            step=0
            current_positions = [maze_start for i in range(len(chromosomes))]
            visited = [[maze_start] for i in range(len(chromosomes))]
            auto=False
                


            pygame.time.delay(100)
            

        

        if draw:
            #add visited points to visited array
            for i in range(len(current_positions)):
                visited[i].append(current_positions[i])

        

     

            

        pygame.display.update()

    
    #close pygame window
    pygame.quit()


def doit(filename):

    print("Starting",filename)

   
    #set some default pygad parameters
    params = {
        "num_generations": 200,
        "population_size": 100,
        "crossover_func": "custom",
        "mutation_func": "custom",
        "mutation_probability": 0.1,
        "population_parents_percent":  0.05, #how many parents we will use for next generation (0.05 means 5%) for 200 population we will have 10 parents
        "population_keep_elite_percent": 0.01, #how many elite we will use for next generation (0.05 means 5%) for 200 population we will have 10 elite       
        "simulate_population": False,
        "crossover_type": "",
        "show_progress": False,
        "smart": False,
        "plot": False,
        "save_to_file": True,
        "maze_file": filename,
        
        }

    
    #crate a Solver new solver object instance
    solver = Solve.Solve(filename,params)
    
    

    filename, maze,maze_start,maze_end,solution, solution_fitness, solution_idx = solver.run()
    print("solution: ",filename, solution)
    print("solution_fitness: ",filename, solution_fitness)
    print("solution_idx: ",filename,solution_idx)


    return filename,maze,maze_start,maze_end,solution

    #simulate best chromosome


    




if __name__ == '__main__':

    import os

    #get all files in folder
    files = os.listdir("mazes")

    #only get one
    files = ["mazes/maze_5.txt"]


    #use ThreadPoolExecutor to run multiple mazes at the same time and get results

    with ThreadPoolExecutor() as executor:
        results = executor.map(doit, files)
        #wait for all results
        results = list(results)

        






       
            
        
           




    #list all files in the folder
   
        
        







