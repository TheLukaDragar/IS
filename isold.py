#main 
import numpy as np

#pygad
import pygad

#imort plit 
import matplotlib.pyplot as plt

import pygame
import time



maze_np=None
start=None
end=None






def read_maze(filename,maze_num):
    #mazes are separated by a blank line
    #maze_num is the number of the maze to read

    #open the file
    f=open(filename,"r")

    #read the maze
    maze=[]

    #read the maze
    for line in f:
        if line=="\n":
            maze_num-=1
            if maze_num<0:
                break
        else:
            #read the maze line by line
            maze.append(line.strip())

    #close the file
    f.close()


    return maze


def make_maze(maze):

    #maze is 2d numpy array of 0-1 where 0 is a wall and 1 is a path


    start=None
    end=None

    maze_np = np.zeros((len(maze),len(maze)),dtype=int)
    print(maze_np.shape)

    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j]=="S":
                start=(i,j)
                maze_np[i][j]=1
            if maze[i][j]=="E":
                end=(i,j)
                maze_np[i][j]=1
            if maze[i][j]=="#":
                maze_np[i][j]=0
            if maze[i][j]==".":
                maze_np[i][j]=1

    return start,end,maze_np




def distance_from_end(path,end):
    #compute the distance from the end

    return np.sqrt((path[-1][0]-end[0])**2+(path[-1][1]-end[1])**2)
    

def get_walls_hit(path,maze_np,fast=False):

    #path is a list of coordinates
    #maze_np is a numpy array of 0-1 where 0 is a wall and 1 is a path

    #return the number of times the path goes through a wall

    walls_hit=0

    for i in range(len(path)):
        #check if the path is out of the maze
        if path[i][0]<0 or path[i][0]>=maze_np.shape[0] or path[i][1]<0 or path[i][1]>=maze_np.shape[1]:
            walls_hit+=1
            if fast:
                return walls_hit

        else:
            if maze_np[path[i][0]][path[i][1]]==0:
                walls_hit+=1
                if fast:
                    return walls_hit

    return walls_hit


def get_same_place_penalty(path):
    #path is a list of coordinates
    #return the number of times the path goes through the same place

    same_place_penalty=0

    for i in range(len(path)):
        for j in range(i+1,len(path)):
            if path[i]==path[j]:
                same_place_penalty+=1

    return same_place_penalty


def get_path(solution,start):
    path=[start]
    pos=start

    for i in range(len(solution)):
        if solution[i]==0:
            pos=(pos[0]-1,pos[1])
        if solution[i]==1:
            pos=(pos[0]+1,pos[1])
        if solution[i]==2:
            pos=(pos[0],pos[1]-1)
        if solution[i]==3:
            pos=(pos[0],pos[1]+1)

        path.append(pos)

    return path


    
def get_out_of_maze_penalty(path,maze_np):
    #path is a list of coordinates
    #maze_np is a numpy array of 0-1 where 0 is a wall and 1 is a path

    #return the number of times the path goes out of the maze

    out_of_maze_penalty=0

    for i in range(len(path)):
        if path[i][0]<0 or path[i][0]>=maze_np.shape[0] or path[i][1]<0 or path[i][1]>=maze_np.shape[1]:
            out_of_maze_penalty+=1

    return out_of_maze_penalty


def get_back_and_forth_penalty(path):
    #path is a list of coordinates
    #return the number of times the path goes back and forth
   



    back_and_forth_penalty=0

    for i in range(len(path)-1):
        if path[i]==path[i+1]:
            back_and_forth_penalty+=1

    

    return back_and_forth_penalty


def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        #random_split_point = np.random.choice(range(offspring_size[1]))

        #parent1[random_split_point:] = parent2[random_split_point:]

        #crossover function must not create a path that goes out of the maze or through a wall
        #so we need to check the path and if it is not valid, we need to try again

        path_par1=get_path(parent1,start)
        path_par2=get_path(parent2,start)

        #combining them at points in the graph they have in common and find valid path
        #if they have no common points, then just use parent1
        common_points=[]
        valid_path=False

        while not valid_path:
            for i in range(len(path_par1)):
                for j in range(len(path_par2)):
                    if path_par1[i]==path_par2[j]:
                        common_points.append(path_par1[i])

            if len(common_points)==0:
                valid_path=True
                parent1=path_par1
            else:
                #find the first common point
                common_point=common_points[0]
                for i in range(len(path_par1)):
                    if path_par1[i]==common_point:
                        break

                for j in range(len(path_par2)):
                    if path_par2[j]==common_point:
                        break

                #combine the paths
                parent1=path_par1[:i]+path_par2[j:]

                #check if the path is valid
                if get_walls_hit(parent1,maze_np,fast=True)==0:
                    valid_path=True

                #clear the common points
                common_points=[]
            
        
        offspring.append(parent1)

        idx += 1

    return np.array(offspring)



def mutation_func(offspring, ga_instance):

    #mutation function must not create a path that goes out of the maze or through a wall


    #check if the path is valid
    #if get_walls_hit(path,maze_np,fast=True)==0:
        

       

    return offspring

def get_valid_directions(pos,maze_np):
    up=0
    down=0
    left=0
    right=0

    if pos[0]-1>=0:
        if maze_np[pos[0]-1][pos[1]]==1:
            up=1

    if pos[0]+1<maze_np.shape[0]:
        if maze_np[pos[0]+1][pos[1]]==1:
            down=1

    if pos[1]-1>=0:
        if maze_np[pos[0]][pos[1]-1]==1:
            left=1

    if pos[1]+1<maze_np.shape[1]:
        if maze_np[pos[0]][pos[1]+1]==1:
            right=1

    return up,down,left,right

def get_next_pos(pos,direction):
    if direction==0:
        return (pos[0]-1,pos[1])
    if direction==1:
        return (pos[0]+1,pos[1])
    if direction==2:
        return (pos[0],pos[1]-1)
    if direction==3:
        return (pos[0],pos[1]+1)


def main():
    maze =read_maze("maze.txt",4)
    #turn maze into a numpy array and convert to integers and get S-start and E-end
    for i in range(len(maze)):
        print(maze[i])

    global start
    global end
    global maze_np
    

    start,end,maze_np=make_maze(maze)
    

    print(start,end)
    assert maze[start[0]][start[1]]=="S"
    #print(maze[start[0]][start[1]])
    assert maze[end[0]][end[1]]=="E"
    #pretty print the maze
    for i in range(len(maze_np)):
        print(maze_np[i])

    #use pygad to solve the maze
    #pygad is a genetic algorithm library

    #define the fitness function
    def fitness_func(solution, solution_idx):
        #fitness is a weighted sum of the distance from the end and the number off times the path goes through a wall (penalty) 
        #and the number of times the path goes through the same place (penalty) 

        #solution is a numpy array of 0-3 where 0 is up, 1 is down, 2 is left, 3 is right


       
        
        #transform solution into coordinates
        path=get_path(solution,start)

        distance=distance_from_end(path,end)
        
        #smaller distance is better
        distance=1/distance**2+0.0000001

        #walls_hits = get_walls_hit(path,maze_np) * 8888888
        #same place penalty
        #same_place=0
        #get out of maze penalty
        #out_of_maze_penalty=get_out_of_maze_penalty(path,maze_np) * 99999999

        #penalty for going back and forth
        #back_and_forth_penalty=0

        #scale the distance based on the number of walls hit and the number of times the path goes through the same place\
        #and the number of times the path goes out of the maze
        #fitness=distance/(walls_hits+same_place+out_of_maze_penalty+back_and_forth_penalty+1)

        fitness=distance

        



        
            

        return fitness




    #define the number of genes this is the number of steps to take
    num_genes=len(maze_np)*len(maze_np)

    #define the number of parents this is the number of solutions that will be used to create the next generation
    num_parents_mating=10

    #define the number of generations
    num_generations=100

    #define the mutation probability
    mutation_percent_genes=10

    #create the initial population



    #create an instance of the genetic algorithm
    # with mutation_percent_genes=10 the mutation probability is 10%
    # with num_parents_mating=10 the number of parents that will be used to create the next generation is 10
    # with num_generations=100 the number of generations is 100
    # with num_genes=len(maze_np)*len(maze_np) the number of genes is the number of steps to take
    # with fitness_func=fitness_func the fitness function is fitness_func
    # with init_range_low=0 the lowest value of a gene is 0
    # with init_range_high=3 the highest value of a gene is 3
    # with parent_selection_type="sss" the parent selection type is stochastic sampling selection
    # with keep_parents=1 the number of parents that will be kept in the next generation is 1
    #sol_per_pop=100 the number of solutions in the population is 100
    # with random_mutation_min_val=0 the lowest value of a gene after mutation is 0
    # with random_mutation_max_val=3 the highest value of a gene after mutation is 3
    #gene_type=int the type of a gene is integer

    #generate 100 valid paths
    initial_population=[]
    for i in range(20):

        path=np.zeros(num_genes,dtype=int)
        current_position=start

        for j in range(num_genes):
            #chose random direction
           
            up,down,left,right=get_valid_directions(current_position,maze_np)
            assert up+down+left+right!=0
            #chose one of the valid directions
            direction=np.random.choice(range(4))
            while True:
                if direction==0 and up==1:
                    break
                elif direction==1 and down==1:
                    break
                elif direction==2 and left==1:
                    break
                elif direction==3 and right==1:
                    break
                else:
                    direction=np.random.choice(range(4))

                #add next poin as last point

            current_position=get_next_pos(current_position,direction)
            path[j]=direction
            
        initial_population.append(path)


    initial_population=np.array(initial_population)

    print("Generrated initial population")





    ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_func,
                            sol_per_pop=20,
                            num_genes=num_genes,
                            init_range_low=0,
                            init_range_high=3,
                            mutation_percent_genes=mutation_percent_genes,
                            parent_selection_type="sss",
                            keep_parents=1,
                            crossover_type=crossover_func,
                            mutation_type=mutation_func,
                            initial_population=initial_population,
                            random_mutation_min_val=0,
                            random_mutation_max_val=3,
                            gene_type=int)




    #run the genetic algorithm
    ga_instance.run()

    #get the best solution
    solution, solution_fitness, solution_idx=ga_instance.best_solution()

    #print the best solution
    print("Best solution:")
    print(solution)
    print("Best solution fitness:")
    print(solution_fitness)
    #check if the solution is correct
    path=get_path(solution,start)
    print("Path:")
    print(path)
    print("Distance from end:")
    print(distance_from_end(path,end))
    print("Walls hit:")
    print(get_walls_hit(path,maze_np))
    print("Same place penalty:")
    print(get_same_place_penalty(path))
    print("Out of maze penalty:")
    print(get_out_of_maze_penalty(path,maze_np))

    #plot the fitness of the best solution
    # plt.plot(ga_instance.best_solutions_fitness)
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.show()

    #usepygame to display the maze and the path
    pygame.init()
    #
    screen = pygame.display.set_mode((len(maze_np)*100,len(maze_np)*100))
    #add a text to the screen
    font = pygame.font.SysFont('Comic Sans MS', 30)
    text_surface = font.render('Best solution fitness: '+str(solution_fitness), False, (0, 0, 0))
    screen.blit(text_surface,(0,0))

    pygame.display.set_caption("Maze")
    clock = pygame.time.Clock()
    running = True

    steps=0
    auto = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0,0,0))
        for i in range(len(maze_np)):
            for j in range(len(maze_np)):
                if maze_np[i][j]==1:
                    #scale the maze to fit the screen
                    pygame.draw.rect(screen,(255,255,255),(j*10,i*10,10,10))
        
        
        pygame.draw.rect(screen,(0,255,0),(start[1]*10,start[0]*10,10,10))
        pygame.draw.rect(screen,(255,0,0),(end[1]*10,end[0]*10,10,10))
        
       

       #draw the current position
        pygame.draw.rect(screen,(0,0,255),(path[steps][1]*10,path[steps][0]*10,10,10))
    

        

        pygame.display.flip()

        #wait for key press before moving to the next step W

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    steps+=1

                    if auto:
                        auto=False
                    #print steps and coordinates of the current step
                    #format: step=1, coordinates=(0,0)
                    print("step="+str(steps)+", coordinates="+str(path[steps-1]))

                    break

                #r is for reset
                if event.key == pygame.K_r:
                    steps=0
                    break

                #a is for auto
                if event.key == pygame.K_a:
                    auto=True
                    break
            if event.type == pygame.QUIT:
                running = False

        if auto:
            steps+=1
            print("step="+str(steps)+", coordinates="+str(path[steps-1]))
            if steps==len(path):
                steps=0
                auto=False
            #clock.tick(1)
            time.sleep(0.1)


        if steps==len(path):
            break

        #




    pygame.quit()


        
        
        
       


        



    
    

    

    

    


    





















if __name__ == '__main__':
    main()