import argparse
import base64
import itertools
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from random import randint, seed
import threading

import matplotlib.pyplot as plt
import numpy as np
import pygame as pygame
from numpy import arange

import test2 as t2
import test3 as t3
from is4 import IS4
import tqdm as tqdm

filename="xx"

lock = threading.Lock()
pbar = None


maze_dict = {
            "#": 0,
            ".": 1,
            "S": 2,
            "E": 3,
            "T": 4,
        }


def do(i):
    t2.dosomethig("test"+str(i))

def do2(combination):

    #unpack the combination

    population_size, population_parents_percent, mutation_probability, population_func, crossover_type, smart, parent_selection_type = combination



    global filename
    is4=IS4()
    is4.read_maze("mazes/"+filename)
    params = {
        "num_generations": 800,
        "population_size": population_size,
        "crossover_func": "custom",
        "mutation_func":  "custom",
        "mutation_probability": mutation_probability,
        # how many parents we will use for next generation (0.05 means 5%) for 200 population we will have 20 parents
        "population_parents_percent":  population_parents_percent,
        # how many elite we will use for next generation (0.05 means 5%) for 200 population we will have 10 elite
        "population_keep_elite_percent": 0.01,
        "population_func": population_func,  # invalid, valid, valid_smart
        "simulate_population": False,
        "crossover_type": crossover_type,
        "show_progress": False,
        "smart":smart, #if mutation is smart 
        "show_plot": False,
        "self_save_to_file": False,
        "maze_file": filename,
        "parent_selection_type": parent_selection_type,
        "seed": 100,
    }

    #print(filename+" starting"+str(worker_id))
    maze, maze_start, maze_end, solution, solution_fitness, solution_idx,best_list = is4.start_ga(params)
    #print(filename+" done!")
    #print("best fitness",solution_fitness)
    #print("best solution",solution)
    #print("best solution index",solution_idx)
    #get the max value index of the list from left to right
    max_index = best_list.index(max(best_list))
    #print("needed generations for best",max_index,"/",params["num_generations"])

    #print("----------------------------------------")

    #lock
    lock.acquire()

    #update pbar 
    pbar.update(1)

    #unlock
    lock.release()


    return filename,best_list,solution,solution_fitness,solution_idx,max_index,params,maze,maze_start



def simulate_chromosomes(maze,maze_start,chromosomes, title="", auto=True, draw=False, custom_colors=[], crossover_data=None, auto_stop=False):
        # use pygame to simulate chromosome is a numpy array of moves

        # open pygame window
        pygame.init()
        #RESizable

        #get computer screen size
        infoObject = pygame.display.Info()
        screen_width = 800
        screen_height = 800


        maze_size_x = maze.shape[1]
        maze_size_y = maze.shape[0]

        #we must fit maze in window of size display_size and keep aspect ratio
        #we will use min of display_size[0] / maze_size_x and display_size[1] / maze_size_y
        #to calculate cell_size

        cell_size = min(screen_width // maze_size_x, screen_height // maze_size_y)

        #we must center maze in window
        #we will use display_size[0] - cell_size * maze_size_x and display_size[1] - cell_size * maze_size_y
        #to calculate offset

        offset_x = (screen_width - cell_size * maze_size_x) // 2
        offset_y = (screen_height - cell_size * maze_size_y) // 2
        

        #we must calculate font size

        screen=pygame.display.set_mode((screen_width, screen_height),)


        pygame.display.set_caption(title)
        clock = pygame.time.Clock()

        game_finished = False
        running = True
        step = 0
        current_positions = [maze_start for i in range(len(chromosomes))]
        colors = custom_colors if custom_colors else [(randint(0, 255), randint(
            0, 255), randint(0, 255)) for i in range(len(chromosomes))]

        # aray of arrays of visited points
        visited = [[maze_start] for i in range(len(chromosomes))]

        if crossover_data:
            # we have crossover data so we draw the crossover point
            intersection_point = crossover_data["intersection_point"]
            intersection_point_chromosome1_index = crossover_data["c1_rand_intersection_index"]
            intersection_point_chromosome2_index = crossover_data["c2_rand_intersection_index"]

        

        # must be different colors for each chromosome\
        while running:

            # draw maze
            for i in range(maze.shape[0]):

                for j in range(maze.shape[1]):
                    if maze[i, j] == maze_dict["#"]:
                        pygame.draw.rect(screen, (0, 0, 0),
                                            (offset_x + j * cell_size, offset_y + i * cell_size, cell_size, cell_size))
                    elif maze[i, j] == maze_dict["S"]:
                        pygame.draw.rect(screen, (0, 255, 0),
                                            (offset_x + j * cell_size, offset_y + i * cell_size, cell_size, cell_size))
                    elif maze[i, j] == maze_dict["E"]:
                        pygame.draw.rect(screen, (255, 0, 0),
                                            (offset_x + j * cell_size, offset_y + i * cell_size, cell_size, cell_size))
                    elif maze[i, j] == maze_dict["T"]:
                        pygame.draw.rect(screen, (255, 255, 0),
                                            (offset_x + j * cell_size, offset_y + i * cell_size, cell_size, cell_size))
                    else:
                        pygame.draw.rect(screen, (255, 255, 255),
                                            (offset_x + j * cell_size, offset_y + i * cell_size, cell_size, cell_size))

            size = 20

            if draw:
                for i in range(len(visited)):
                    for j in range(len(visited[i])):

                        size=cell_size/2

                        # make color more transparent and smaaller by factor of 2 and draw on center of cell
                        colorr = (colors[i][0], colors[i][1], colors[i][2], 100)
                        rect = pygame.Rect(offset_x + visited[i][j][1] * cell_size + cell_size/2 - size/2, offset_y + visited[i][j][0] * cell_size + cell_size/2 - size/2, size, size)
                        pygame.draw.rect(screen, colorr, rect)                       

            # draw current positions with different colors

            for i in range(len(current_positions)):
                # make each one smaller so we can see them
                size = cell_size - 2 * i
                pygame.draw.rect(screen, colors[i], (current_positions[i][1] * cell_size + offset_x, current_positions[i][0] * cell_size + offset_y, size, size))

            # draw crossover data
            if crossover_data:
                if step >= intersection_point_chromosome1_index or step >= intersection_point_chromosome2_index:
                    # draw intersection point yellow
                    size = cell_size // 2
                    pygame.draw.rect(screen, (255, 255, 0), (intersection_point[1] * cell_size + offset_x, intersection_point[0] * cell_size + offset_y, size, size))
                if auto:
                    # stop auto when intersection point is reached
                    if step == intersection_point_chromosome1_index or step == intersection_point_chromosome2_index:
                        auto = False

                        # check if point is in both visited arrays
                        if intersection_point in visited[0] and intersection_point in visited[1]:
                            print("Intersection point is in both visited arrays")
                        else:
                            print("Intersection point is not in both visited arrays")

            # wait for user input
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

                    # if auto is true we will simulate chromosome automatically
                    # if auto is false we will simulate chromosome step by step
                    # of w is pressed we will make a step
                    if event.key == pygame.K_w:
                        auto = False

                        for i in range(len(current_positions)):
                            current_positions[i], _ = make_move(
                                chromosomes[i][step], current_positions[i], maze, maze_dict)
                        step += 1

                    # q is pressed we will quit
                    if event.key == pygame.K_q:
                        running = False
                        pygame.quit()
                        return

                    # reset
                    if event.key == pygame.K_r:
                        current_positions = [
                            maze_start for i in range(len(chromosomes))]
                        step = 0
                        visited = [[maze_start] for i in range(len(chromosomes))]

                





            if auto and not game_finished:
                # make move
                for i in range(len(current_positions)):
                    current_positions[i], _ = make_move(
                        chromosomes[i][step], current_positions[i], maze, maze_dict)
                step += 1

                # delay
                pygame.time.delay(50)

            if step >= len(chromosomes[0]):
                # we reached the end of chromosome
                if auto_stop:
                    auto = False
                    game_finished = True
                    running = False

                else:

                    game_finished = False
                    step = 0
                    current_positions = [
                        maze_start for i in range(len(chromosomes))]
                    visited = [[maze_start] for i in range(len(chromosomes))]
                    auto = True

                pygame.time.delay(10)

            if draw:
                # add visited points to visited array
                for i in range(len(current_positions)):
                    visited[i].append(current_positions[i])

            pygame.display.update()

        # close pygame window
        pygame.quit()

def make_move(move_direction, current_pos, maze,maze_dict):
        # we move in the maze and check if move is valid
        # if move is valid we return new position and valid flag
        # if move is not valid we return current position and invalid flag
        new_pos = current_pos

        if move_direction == 0:
            # up
            new_pos = (current_pos[0] - 1, current_pos[1])
        elif move_direction == 1:
            # down
            new_pos = (current_pos[0] + 1, current_pos[1])

        elif move_direction == 2:
            # left
            new_pos = (current_pos[0], current_pos[1] - 1)

        elif move_direction == 3:
            # right
            new_pos = (current_pos[0], current_pos[1] + 1)

        # check if move is valid
        if new_pos[0] < 0 or new_pos[0] >= maze.shape[0] or new_pos[1] < 0 or new_pos[1] >= maze.shape[1]:
            # move is not valid because we are out of maze
            return current_pos, False

        if maze[new_pos[0], new_pos[1]] == maze_dict["#"]:
            # move is not valid because we hit a wall
            return current_pos, False

        #move is valid
        return new_pos, True



def main(filenameee,worker_i):

    global filename

    filename=filenameee

    parameters = {

        "population_size": list(range(25, 251, 50)),
        
        "population_parents_percent": arange(0.02, 0.20, 0.10),
        "mutation_probability": arange(0.05, 0.20, 0.05),

        "population_func" : ["invalid","valid","valid_smart"],
        "crossover_type":[ "min_max","longest_path_mix","rand_rand"],
        "smart": [True, False], #smart crossover
        "parent_selection_type": ["sus","rws","random"],


    }

    combinations = list(itertools.product(*parameters.values()))
    

    #this worker does combinations from 108*worker_i to worker_i+1 *108
    #2160 combinations

    combinations=combinations[108*worker_i:108*(worker_i+1)]
    combinations_len = len(combinations)

    #20/5=4
    #0-4
    #4-8
    #8-12
    #12-16
    #16-20

   



    #print("Number of combinations:", combinations_len)


    #use tqdm to show progress bar
    
    #inicialize progress bar
    global pbar
    pbar = tqdm.tqdm(total=108, desc="Worker "+str(worker_i), position=worker_i)
    #show progress bar
    pbar.update(0)

    

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(do2, combinations)
        


        #wait for all results

        #check  for results
        for i,result in enumerate(results):

            #plot the results
            _filename,best_list,solution,solution_fitness,solution_idx,max_index,params,maze,maze_start = result


            #reset the plot
            plt.clf()

            plt.plot(best_list, label=filename)
            plt.legend()
            #plt.show(block=True)

            #make dir
            
            #timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            #save to /results/filename/timestamp

            #generate  unique filename
    
            
            
            _filename = _filename + "_" + str(i)
            _filename = _filename + "_" + str(int(solution_fitness))
            _filename = _filename + "_" + str(worker_i)
            #txt
            filename_txt = _filename + ".txt"

            
            

            dir = "results/mazes/"+params["maze_file"]+"/" + _filename
            #check if dir exists
            if not os.path.exists(dir):
                os.makedirs(dir)

        
            

            #save plot to file
            plt.savefig(dir+"/plot.png")
            # close progress bar
            # save to file with name beeing filename and date
        
            # get only filename
            filename = os.path.basename(params["maze_file"])

            
            with open(dir+"/res.txt", "w") as myfile:
                myfile.write(str(solution_fitness) + "\n")
                myfile.write(str(max_index) + "\n") #genrations required
                myfile.write(str("".join(solution.astype(str))) + "\n")
                myfile.write(str(params) + "\n")
                # save best solution fitness
                myfile.write(str(best_list) + "\n")

            #simulator
            #simulate_chromosomes(maze,maze_start,[solution], draw=True, auto_stop=False, crossover_data=None)





if __name__ == '__main__':

    #get arguments from command line

    seed(100)

    #parser arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-filename', type=str, help='filename')
    parser.add_argument('-worker', type=int, help='worker number')

    args = parser.parse_args()

    filename=args.filename
    worker=args.worker

    #print("filename:",filename)
    #print("worker:",worker)

    main(filename,worker)


    




    
    


    
    #filename=["big_maze.txt","maze_1.txt","maze_2.txt","maze_3.txt","maze_4.txt","maze_5.txt","maze_6.txt","maze_7.txt","maze_harder_0.txt","maze_harder_1.txt","maze_harder_10.txt","maze_harder_11.txt","maze_harder_12.txt","maze_harder_13.txt","maze_harder_14.txt","maze_harder_15.txt","maze_harder_16.txt","maze_harder_17.txt","maze_harder_2.txt","maze_harder_3.txt","maze_harder_4.txt","maze_harder_5.txt","maze_harder_6.txt","maze_harder_7.txt","maze_harder_8.txt","maze_harder_9.txt","maze_treasure.txt","maze_treasure_2.txt","maze_treasure_3.txt","maze_treasure_4.txt","maze_treasure_5.txt","maze_treasure_6.txt","maze_treasure_7.txt","mega.txt" 
    #]

    #get only 2



    #2160 combinations
    #if i use 20 workers that 2160/20 = 108 combinations per worker
    #each worker will take combinations[i*108:(i+1)*108] combinations
    
    

    