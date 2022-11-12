import time as t
import pygad

import random as r


h=0

def fitness_func(solution, solution_idx):
   
    return 1

def dosomethig(name):
    print("I am doing something")


    #launch pytgad

    hain = pygad.GA(num_generations=5, num_parents_mating=5, fitness_func=fitness_func,
    sol_per_pop=10, num_genes=10, init_range_low=0, init_range_high=10, mutation_percent_genes=10)

    hain.run()

    global h
    
    #get random number
    
    


    #get results

    solution, solution_fitness, solution_idx = hain.best_solution()


    #print("solution: ", solution)

    if name == "test5":
        print("i am test5 i change h")
        h += r.randint(0,100)
        

    print(name,"h", h)
    
    






    for i in range(100):
        #print(name,i)
        t.sleep(0.5)

    


if __name__ == "__main__":
    dosomethig("me")
    print("I am done")