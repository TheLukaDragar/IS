import numpy as np
import matplotlib.pyplot as plt
import pygad
import pygame
import os
from random import choice, choices, random, shuffle, randint, seed
import sys

from tqdm import tqdm
maze = None
maze_end = None
maze_start = None
num_of_treasures = None
points_of_treasures = set()
crossover_type = "longest_path_mix"

pbar = None


maze_dict = {
    "#": 0,
    ".": 1,
    "S": 2,
    "E": 3,
    "T": 4,
}


def read_maze(file_name):
    global maze, maze_end, maze_start, num_of_treasures
    num_of_treasures = 0

    # read maze from file
    # maze is encoded in the following way:
    # # - wall
    # . - path
    # S - start
    # E - end
    # T - treasure

    # we will use 0 for wall, 1 for path, 2 for start, 3 for end, 4 for treasure

    with open(file_name, 'r') as f:
        # read maze and encode it
        maze = [[maze_dict[c] for c in line.strip()] for line in f.readlines()]

        # find start and end and count treasures
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if maze[i][j] == maze_dict["S"]:
                    maze_start = (i, j)
                elif maze[i][j] == maze_dict["E"]:
                    maze_end = (i, j)
                elif maze[i][j] == maze_dict["T"]:
                    points_of_treasures.add((i, j))
                    num_of_treasures += 1

        assert maze_start is not None
        assert maze_end is not None

        # convert maze to numpy array
        maze = np.array(maze)

    return maze, maze_start, maze_end


def mutate_one_chromosome(chromosome, ga_instance):

    global maze_start

    current_pos = maze_start
    assert current_pos is not None

    # loop through all genes and mutate them
    for gene_idx in range(len(chromosome)):

        valid = []
        for i in range(4):
            new_pos, valid_move = make_move(i, current_pos)
            if valid_move:
                valid.append(i)

        if random() < ga_instance.mutation_probability:
            # we mutate gene
            chromosome[gene_idx] = choice(valid)

        # make moves and fix chromosome if its not valid from previous move
        current_pos, valid_move = make_move(chromosome[gene_idx], current_pos)
        if not valid_move:
            chromosome[gene_idx] = choice(valid)
            current_pos, valid_move = make_move(
                chromosome[gene_idx], current_pos)
            #assert valid_move

    return chromosome

    #

    return chromosome


def mutation_func(offspring, ga_instance):
    offsprings = np.copy(offspring)
    # loop thour all offsprings of parents can be more than 2

    for chromosome_idx in range(offspring.shape[0]):

        offsprings[chromosome_idx] = mutate_one_chromosome(
            offspring[chromosome_idx], ga_instance)

    return offsprings



def crossover_two_chromosomes(chromosome1, chromosome2, sim=False):
    # we combine two chromosomes at the intersection points we choose randomly
    # intersection points have different index in both chromosomes
    # so we choose random index for each chromosome and we combine chromosomes from and to that index

    chromosome1_points = {
        maze_start: [0],
    }
    # this holds all the point the chromosome1 visits and the index of the move that visits that point

    chromosome2_points = {
        maze_start: [0],
    }

    current_pos = maze_start

    for i, direction in enumerate(chromosome1):
        current_pos, _ = make_move(direction, current_pos)
        if current_pos not in chromosome1_points:
            chromosome1_points[current_pos] = [i+1]

        else:
            # append index of move that visits this point
            chromosome1_points[current_pos].append(i+1)

    current_pos = maze_start

    for i, direction in enumerate(chromosome2):
        current_pos, _ = make_move(direction, current_pos)
        if current_pos not in chromosome2_points:
            chromosome2_points[current_pos] = [i+1]

        else:
            chromosome2_points[current_pos].append(i+1)

    # we have all the points that are visited by both chromosomes
    # we need to choose random intersection point
    intersection_points = list(
        set(chromosome1_points.keys()) & set(chromosome2_points.keys()))
    if len(intersection_points) == 0:
        # no intersection points
        # we return one of the parents
        return choice([chromosome1, chromosome2])

    # we have intersection points
    # we choose random intersection point
    intersection_point = choice(intersection_points)

    if crossover_type == "most_visited":
        # we choose the most visited intersection point
        intersection_point = max(
            intersection_points, key=lambda x: len(chromosome1_points[x]))


    # we choose random index of move that visits the intersection point
    intersection_point_chromosome1_index = choice(
        chromosome1_points[intersection_point])
    intersection_point_chromosome2_index = choice(
        chromosome2_points[intersection_point])
        
    if crossover_type == "min_max":
        if sim:
            print("Using min_max")
        #we chose the min index of first chromosome and max index of second chromosome 
        intersection_point_chromosome1_index= min(chromosome1_points[intersection_point])
        #remove this index from chromosome2 
        #check if more than one index is present
        if len(chromosome2_points[intersection_point])>1:
            chromosome2_points[intersection_point].remove(intersection_point_chromosome2_index)
            

        intersection_point_chromosome2_index= max(chromosome2_points[intersection_point])

        

    

    if (crossover_type == "longest_path_mix"):

        c1_l = chromosome2[:intersection_point_chromosome1_index]
        c1_r = chromosome2[intersection_point_chromosome1_index:][::-1]
        c2_l = chromosome1[:intersection_point_chromosome2_index]
        c2_r = chromosome1[intersection_point_chromosome2_index:][::-1]

        new_chromosome = np.concatenate(
            (chromosome1[:intersection_point_chromosome1_index], chromosome2[intersection_point_chromosome2_index:]))

        # find the combintion that is the longest
        c1_c2 = len(np.concatenate(
            (chromosome1[:intersection_point_chromosome1_index], chromosome2[intersection_point_chromosome2_index:])))
        c1_c2_rtl = len(np.concatenate(
            (chromosome1[:intersection_point_chromosome1_index], chromosome2[:intersection_point_chromosome2_index][::-1])))
        c2_c1 = len(np.concatenate(
            (chromosome2[:intersection_point_chromosome2_index], chromosome1[intersection_point_chromosome1_index:])))
        c2_c1_rtl = len(np.concatenate(
            (chromosome2[:intersection_point_chromosome2_index], chromosome1[:intersection_point_chromosome1_index][::-1])))
        if sim:
            print("c1_c2=", (len(c1_l), len(c2_r)), "=", c1_c2)
            print("c1_c2_rtl=", (len(c1_l), len(c2_l)), "=", c1_c2_rtl)
            print("c2_c1=", (len(c2_l), len(c1_r)), "=", c2_c1)
            print("c2_c1_rtl=", (len(c2_l), len(c1_l)), "=", c2_c1_rtl)
            print("-----------------")

        glue_combinations = {
            "c1_c2": c1_c2,
            "c1_c2_rtl": c1_c2_rtl,
            "c2_c1": c2_c1,
            "c2_c1_rtl": c2_c1_rtl,

        }

    # remove all the combinations that are shorter than lenght of the genome
        for key in list(glue_combinations.keys()):
            if glue_combinations[key] < len(chromosome1):
                del glue_combinations[key]

    # we have all the combinations that are longer than the genome
    # we choose the random one
        glue_combination = choice(list(glue_combinations.keys()))

        if glue_combination == "c1_c2":
            new_chromosome = np.concatenate(
                (chromosome1[:intersection_point_chromosome1_index], chromosome2[intersection_point_chromosome2_index:]))
        elif glue_combination == "c1_c2_rtl":
            new_chromosome = np.concatenate(
                (chromosome1[:intersection_point_chromosome1_index], chromosome2[:intersection_point_chromosome2_index][::-1]))
        elif glue_combination == "c2_c1":
            new_chromosome = np.concatenate(
                (chromosome2[:intersection_point_chromosome2_index], chromosome1[intersection_point_chromosome1_index:]))
        elif glue_combination == "c2_c1_rtl":
            new_chromosome = np.concatenate(
                (chromosome2[:intersection_point_chromosome2_index], chromosome1[:intersection_point_chromosome1_index][::-1]))
        if sim:
            print("randomly chose glue_combination=", glue_combination,
                  "with length=", len(new_chromosome))

    else:
        new_chromosome = np.concatenate(
            (chromosome1[:intersection_point_chromosome1_index], chromosome2[intersection_point_chromosome2_index:]))

    # check if the new chromosome if too long
    if len(new_chromosome) > len(chromosome1):
        # we need to cut it
        new_chromosome = new_chromosome[:len(chromosome1)]

    # we combine chromosomes from and to the intersection point
    #new_chromosome= np.concatenate((chromosome1[:intersection_point_chromosome1_index], chromosome2[intersection_point_chromosome2_index:]))
    len_new = new_chromosome.shape[0]
    # we make sure that is long enough else we add random moves that must be valid
    if len(new_chromosome) < len(chromosome1):
        # use make_valid_chromosome to make valid chromosome from the end of the new_chromosome with length len(chromosome1) - len(new_chromosome)
        #assert False, "not implemented any"
        global smart
        new_chromosome = np.concatenate((new_chromosome, make_valid_chromosome(
            from_pos=intersection_point, chromosome_length=len(chromosome1) - len(new_chromosome),smart=True))) # smart=is powerfull

        # save all three chromosomes to file

        # sim=True

        if False:
            with open("crossover_edge_cases.txt", "a") as f:
                f.write("".join([str(x) for x in chromosome1]) + "\n")
                f.write("".join([str(x) for x in chromosome2]) + "\n")
                f.write("".join([str(x) for x in new_chromosome]) + "\n")
                # add intersection point
                f.write(str(intersection_point) + "\n")
                # add intersection points
                f.write(str(intersection_point_chromosome1_index) +
                        " " + str(intersection_point_chromosome2_index) + "\n")

    if len(new_chromosome) > len(chromosome1):
        new_chromosome = new_chromosome[:len(chromosome1)]

    if sim:

        #print("Chromosome 1 points:", chromosome1_points)
        #print("Chromosome 2 points:", chromosome2_points)
        #print("Intersection points:", intersection_points)
        print("Intersection point:", intersection_point)
        print("Intersection point chromosome 1 index:",
              intersection_point_chromosome1_index)
        print("Intersection point chromosome 2 index:",
              intersection_point_chromosome2_index)

        print("1:", "".join([str(x) for x in chromosome1]))
        print("2:", "".join([str(x) for x in chromosome2]))
        print("3:", "".join([str(x) for x in new_chromosome]))
        if len_new < len(chromosome1):
            print("chromosome was too short made it longer by", len(
                chromosome1) - len_new, "from", len_new, "to", len(new_chromosome))

        print("chose intersection point:", intersection_point, "with rand index in both chromosomes(1,2):",
              intersection_point_chromosome1_index, intersection_point_chromosome2_index)
        print("and combined chromosomes from and to that index"+"\nadded chromosome1 part:", "".join(
            [str(x) for x in chromosome1[:intersection_point_chromosome1_index]]), "\nadded chromosome2 part:", "".join([str(x) for x in chromosome2[intersection_point_chromosome2_index:]]))
        toprint = "".join([str(x) for x in chromosome1[:intersection_point_chromosome1_index]]) + \
            " "+"".join([str(x)
                        for x in chromosome2[intersection_point_chromosome2_index:]])
        if len_new < len(chromosome1):
            toprint += " " + \
                "".join([str(x) for x in new_chromosome])[
                    :len(chromosome1) - len_new]
        print("new chromosome:", toprint)

        print("--------------------")

        crossover_data = {
            "c1_rand_intersection_index": intersection_point_chromosome1_index,
            "c2_rand_intersection_index": intersection_point_chromosome2_index,
            "intersection_point": intersection_point,



        }

        # simulate all 3 chromosomes visually using pygame
        # first color is red
        # second color is green
        # third color is blue
        custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        simulate_chromosomes([chromosome1, chromosome2, new_chromosome], title="Crossover",
                             auto=False, draw=True, custom_colors=custom_colors, crossover_data=crossover_data)
        # print the chromosomes points

    return new_chromosome


def crossover_func(parents, offspring_size, ga_instance):
    offspring = np.empty(shape=offspring_size)

    for i in range(offspring_size[0]):
        # we have more than 2 parents so we need to choose 2 parents randomly
        parent1_idx = choice(range(len(parents)))
        parent2_idx = choice(range(len(parents)))

        offspring[i] = crossover_two_chromosomes(
            parents[parent1_idx], parents[parent2_idx])
    return offspring


def make_move(move_direction, current_pos):
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


def make_valid_chromosome(from_pos=None, chromosome_length=None, smart=False):
    # chromosome is a list of moves of size maze_x * maze_y
    # each move is encoded as a number from 0 to 3
    #0 - up
    #1 - down
    #2 - left
    #3 - right

    # we must write a valid chromosome with encuraging taking different paths
    # and not getting stuck in a loop

    # loop for rows * cols times and make a move and save points visited

    visited = {}

    current_pos = from_pos if from_pos else maze_start
    chromosome_length = chromosome_length if chromosome_length else maze.shape[
        0] * maze.shape[1]

    chromosome = []

    for i in range(chromosome_length):
        # make move in every direction and check if move is valid also sort based on visited points
        # we want to avoid loops

        moves = []
        # we make a random sequence of moves to avoid same moves every time
        random_seq = [0, 1, 2, 3]
        shuffle(random_seq)

        for move_direction in random_seq:
            new_pos, valid = make_move(move_direction, current_pos)
            if valid:
                moves.append((move_direction, visited.get(new_pos, 0)))

        # randomly sort moves based on visited points

        if smart:
            moves.sort(key=lambda x: x[1])
        # sort moves based on visited points

        # choose valid move with least visited points
        move_direction, _ = moves[0]

        # make move
        current_pos, _ = make_move(move_direction, current_pos)

        # save move
        chromosome.append(move_direction)

        # update visited points
        visited[current_pos] = visited.get(current_pos, 0) + 1

    return chromosome


def make_valid_population(population_size):
    global smart
    population = []
    for i in range(population_size):
        population.append(make_valid_chromosome(smart=smart))

    return population


def make_invalid_population(population_size):
    global smart
    population = []
    for i in range(population_size):
        #generate random chromosome
        chromosome = [randint(0, 3) for _ in range(maze.shape[0] * maze.shape[1])]
        population.append(chromosome)


    return population



def fitness_func(solution, solution_idx):

    # calculate distance from current position to end position
    # we use manhattan distance

    global maze_start

    pos = maze_start

    times_reached_end = 0

    visited = set()

    visited.add(pos)

    found_end = False

    treasures = 0

    success = 1
    steps = 0

    invalid_moves = 0


    for i in range(len(solution)):

        pos, valid = make_move(solution[i], pos)
        p= tuple(pos)
    
        steps += 1

        # check if went back and forth between 2 points and penalize

        if not valid:
            invalid_moves += 1
            #assert False, "Invalid move in chromosome"

        # check if we reached end
        if pos == maze_end and p not in visited:
            found_end = True
            success += 0.10
            visited.add(pos)
            break

        # check if we reached treasure
        if pos in points_of_treasures and p not in visited:
            treasures += 1
            success += 0.15

        visited.add(pos)

    # get number of unique points visited
    unique_points_visited = len(visited)
    # dublicate points visited
    repeated_points_visited = steps - unique_points_visited

    path_left = len(solution) - steps

    # we want to maximize unique points visited and minimize repeated points visited
    # we want to maximize treasures and minimize steps
    # we want to maximize success and minimize path left

    fitness = unique_points_visited

    if found_end and treasures == len(points_of_treasures):

        return fitness + repeated_points_visited*0.9 + path_left*2

    elif found_end and treasures != len(points_of_treasures):
        return (fitness) * (treasures//len(points_of_treasures)+1) - repeated_points_visited*0.05 - path_left*0.01

    return fitness*success - invalid_moves *2

    # we want to minimize distance so we return 1 / distance**2
    return 1 / (distance)+1

    return 1


def simulate_chromosomes(chromosomes, title="", auto=True, draw=False, custom_colors=[], crossover_data=None, auto_stop=False):
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
                            chromosomes[i][step], current_positions[i])
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
                    chromosomes[i][step], current_positions[i])
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


def on_parents(ga_instance, selected_parents):
    # we will use this function to simulate chromosomes
    # we will simulate chromosomes with best fitness

    return

    # get best chromosomes
    best_chromosomes = ga_instance.best_solutions.copy()

    # only simulate last

    if (len(best_chromosomes) % 10 == 0):

        name = "best_chromosome_"+str(len(best_chromosomes))

        # simulate chromosomes all at once
        simulate_chromosomes([best_chromosomes[-1]],
                             title=name, auto_stop=True, draw=True)


def on_generation(ga_instance):
    # add one to tqdm
    if pbar:
        pbar.update(1)

    # todo
    pass


def start_ga(params):
    # get pid
    pid = os.getpid()

    print("Starting GA a!", pid, params["maze_file"])
    # generate initial population

    read_maze(params["maze_file"])

    global smart
    smart = params["smart"] if "smart" in params else False

    #population = make_valid_population(params["population_size"])
    population = make_invalid_population(params["population_size"])

    print("Initial population generated !", params["maze_file"])

    # simulate chromosomes
    if params["simulate_population"]:
        for chromosome in population[0:2]:
            simulate_chromosomes([chromosome])

    # make into numpy array
    population = np.array(population)

    if params["show_progress"]:
        # inicialize progress bar
        global pbar
        pbar = tqdm(total=params["num_generations"])

    if params["mutation_func"] == "custom":
        mutation_func0 = mutation_func
    else:
        mutation_func0 = "random"

    if params["crossover_func"] == "custom":
        global crossover_type
        crossover_type = params["crossover_type"]
        crossover_func0 = crossover_func
    else:
        crossover_func0 = "single_point"

    num_parents_matting = params["population_parents_percent"] * \
        params["population_size"]
    num_parents_matting = int(num_parents_matting)
    # make sure is at least 2
    num_parents_matting = max(2, num_parents_matting)

    num_of_elite_to_keep = params["population_keep_elite_percent"] * \
        params["population_size"]
    # int
    num_of_elite_to_keep = int(num_of_elite_to_keep)

    # sss, rws, tournament
    parent_selection_type = params["parent_selection_type"] if "parent_selection_type" in params else "tournament"

    seddd= params["seed"] if "seed" in params else None
    

    # cast crossover type ftom params to function pointer
    # start genetic algorithm pygad
    ga_instance = pygad.GA(num_generations=params["num_generations"],
                           # number of solutions to be selected as parents in the mating pool.
                           num_parents_mating=num_parents_matting,
                           fitness_func=fitness_func,
                           sol_per_pop=params["population_size"],
                           num_genes=len(population[0]),
                           init_range_low=0,
                           init_range_high=4,  # 0 to 3
                           mutation_by_replacement=mutation_func0 == "random",
                           mutation_type=mutation_func0,
                           crossover_type=crossover_func0,
                           gene_type=int,
                           parent_selection_type=parent_selection_type,
                           keep_elitism=num_of_elite_to_keep,
                           on_generation=on_generation,
                           random_mutation_min_val=0,
                           random_mutation_max_val=4,
                           initial_population=population,
                           parallel_processing=["thread", 256],
                           mutation_probability=params["mutation_probability"],
                           save_best_solutions=True,
                           on_parents=on_parents,
                           random_seed=seddd



                           )

    # run genetic algorithm

    ga_instance.run()

    # get best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # plot fitness with pygad

    if True == params["plot"]:

        # plot fitness
        plt.plot(ga_instance.best_solutions_fitness)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(params["maze_file"], fontsize=14)

        # add sub

        plt.show(block=False)

    if params["save_to_file"]:

        #make dir
        from datetime import datetime
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        #save to /results/filename/timestamp
        dir = "results/"+params["maze_file"]+"/"+timestamp
        os.makedirs(dir)

            


        #save plot to file
        plt.savefig(dir+"/plot.png")
        # close progress bar
        # save to file with name beeing filename and date
       
        # get only filename
        filename = os.path.basename(params["maze_file"])

        
        with open(dir+"/res.txt", "w") as myfile:
            myfile.write(str(solution_fitness) + "\n")
            myfile.write(str("".join(solution.astype(str))) + "\n")
            myfile.write(str(params) + "\n")
            # save best solution fitness
            myfile.write(str(ga_instance.best_solutions_fitness) + "\n")

    return maze, maze_start, maze_end, solution, solution_fitness, solution_idx


def simulatecrossover(typee=""):
    global crossover_type
    crossover_type = typee
    # create a empty maze 100 x 100
    global maze
    maze = np.ones((50, 50), dtype=int)

    # make a wall
    maze[0:50, 0] = 0
    maze[0:50, 49] = 0
    maze[0, 0:50] = 0
    maze[49, 0:50] = 0

    # add start and end
    # star is half way on the left
    maze[25, 1] = 2
    global maze_start
    maze_start = (25, 1)
    # end is half way on the right
    maze[25, 49] = 3
    global maze_end
    maze_end = (25, 49)

    #0 is up
    # 1 down
    # 2 left
    # 3 right

    # make 2 chromosomes
    # they both go right 20 times
    chromosome1 = [3] * 10
    chromosome2 = [3] * 10

    # then one goes up 20 times
    chromosome1 += [0] * 10

    # other goues down 20 times
    chromosome2 += [1] * 10

    # then one goes right 20 times
    chromosome1 += [3] * 10
    # the other one to
    chromosome2 += [3] * 10

    # then one goes down 20 times
    chromosome1 += [1] * 10
    # the other one goes up 20 times
    chromosome2 += [0] * 10

    # then they both go right 20 times to the end
    chromosome1 += [3] * 30
    chromosome2 += [3] * 30

    # do crossover

    c3 = crossover_two_chromosomes(chromosome1, chromosome2, sim=True)


def read_cross_over_events_from_file(event_index):
    # read events from file
    events = []
    with open("crossover_edge_cases.txt", "r") as f:
        # each event is 5 lines
        # first line is chromosome 1
        # second line is chromosome 2
        # third line is chromosome 3
        # fourth line intersection point (x,y)
        # fifth line is random inexes of intersection point of chromosome 1 and chromosome 2
        # read fir
        # read 5 lines at event_index
        for i in range(event_index * 5):
            f.readline()

        # read chromosome 1
        chromosome1 = f.readline()
        chromosome1 = chromosome1.strip()
        # trurn into array of ints
        chromosome1 = [int(x) for x in chromosome1.split("")]
        # read chromosome 2
        chromosome2 = f.readline()
        chromosome2 = chromosome2.strip()
        # trurn into array of ints
        chromosome2 = [int(x) for x in chromosome2.split("")]
        # read chromosome 3
        chromosome3 = f.readline()
        chromosome3 = chromosome3.strip()
        # trurn into array of ints
        chromosome3 = [int(x) for x in chromosome3.split("")]

        # read intersection point
        intersection_point = f.readline()
        intersection_point = intersection_point.strip()
        # tuple is writen as (x, y)
        intersection_point = intersection_point[1:-1]
        intersection_point = intersection_point.split(",")
        intersection_point = (
            int(intersection_point[0]), int(intersection_point[1]))

        # read random indexes
        random_indexes = f.readline()
        # split on " "
        random_indexes = random_indexes.split(" ")
        indx1 = int(random_indexes[0])
        indx2 = int(random_indexes[1])

    return chromosome1, chromosome2, chromosome3, intersection_point, indx1, indx2


def sim_cross_over_events(chromosome1, chromosome2, new_chromosome, intersection_point, intersection_point_chromosome1_index, intersection_point_chromosome2_index):
    print("Intersection point:", intersection_point)
    print("Intersection point chromosome 1 index:",
          intersection_point_chromosome1_index)
    print("Intersection point chromosome 2 index:",
          intersection_point_chromosome2_index)

    print("1:", "".join([str(x) for x in chromosome1]))
    print("2:", "".join([str(x) for x in chromosome2]))
    print("3:", "".join([str(x) for x in new_chromosome]))

    c3 = np.concatenate((chromosome1[:intersection_point_chromosome1_index],
                        chromosome2[intersection_point_chromosome2_index:]))

    print("chromosome was too short made it longer by", len(
        chromosome1) - len, "from", intersection_point_chromosome1_index)

    print("chose intersection point:", intersection_point, "with rand index in both chromosomes(1,2):",
          intersection_point_chromosome1_index, intersection_point_chromosome2_index)
    print("and combined chromosomes from and to that index"+"\nadded chromosome1 part:",
          "".join([str(x) for x in chromosome1[:intersection_point_chromosome1_index]]), "\nadded chromosome2 part:", "".join([str(x) for x in chromosome2[intersection_point_chromosome2_index:]]))
    toprint = "".join([str(x) for x in chromosome1[:intersection_point_chromosome1_index]]) + \
        " "+"".join([str(x)
                    for x in chromosome2[intersection_point_chromosome2_index:]])
    print("new chromosome:", toprint)
    print("--------------------")

    # simulate all 3 chromosomes visually using pygame
    # first color is blue
    # second color is red
    # third color is green
    custom_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    simulate_chromosomes([chromosome1, chromosome2, new_chromosome],
                         title="Crossover", auto=False, draw=True, custom_colors=custom_colors)
    # print the chromosomes points

    return 0

# main


def read_chromosome_from_file(filename):
    # read chromosome from file
    chromosomee = []
    with open(filename, "r") as f:
        chromosome = f.readline()
        chromosome = chromosome.strip()
        for i in chromosome:
            if i == "3":
                chromosomee.append(1)

            elif i == "2":
                chromosomee.append(0)

            elif i == "1":
                chromosomee.append(3)

            elif i == "0":
                chromosomee.append(2)

    return chromosomee


if __name__ == "__main__":
    seed(100)

    # simulate crossover
    # simulatecrossover()
    #c1 = read_chromosome_from_file("chromosome1.txt")
    ##0 - up
    #1 - down
    #2 - left
    #3 - right

    #simulatecrossover("most_visited") # min_max most_visited longest_path_mix
    #exit(0)

    #read_maze(sys.argv[1])
    # simulate chromosome 1
    #simulate_chromosomes([c1],title="Chromosome 1",auto=False,draw=True,custom_colors=[(0,0,255)])

    # get all filenames in folder

    # get params of main

    filename = sys.argv[1]

    # set random seed

    #min_max
    #takes a random intersection point and combines the chromosomes from that point to the end
    #takes the shortest first path and combines it with the "shortest" second path if it is looping it takes part after the loop
    #take part to the loop(if it exists) and combine it with the after loop path of second path

    # set some default pygad parameters
    params = {
        "num_generations": 2000,
        "population_size": 200,
        "crossover_func": "custom",
        "mutation_func":  "custom",
        "mutation_probability": 0.05,
        # how many parents we will use for next generation (0.05 means 5%) for 200 population we will have 20 parents
        "population_parents_percent":  0.20,
        # how many elite we will use for next generation (0.05 means 5%) for 200 population we will have 10 elite
        "population_keep_elite_percent": 0.01,
        "population_func": make_valid_population,
        "fitness_func": fitness_func,
        "simulate_population": False,
        "crossover_type": "min_max",
        "show_progress": True,
        "smart": True,
        "plot": True,
        "save_to_file": True,
        "maze_file": filename,
        "parent_selection_type": "sus",
        "seed": 100,


    }

    maze, maze_start, maze_end, solution, solution_fitness, solution_idx = start_ga(
        params)
    print("solution: ", solution)
    print("solution_fitness: ", solution_fitness)
    print("solution_idx: ", solution_idx)

    # simulate best chromosome

    simulate_chromosomes([solution], draw=True)
