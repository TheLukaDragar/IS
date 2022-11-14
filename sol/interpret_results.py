

import sys
import os

from is4 import IS4

from test import *
#import plt

from matplotlib import pyplot as plt








def main(filename):

    solver = IS4()
    maze,maze_start,maze_end = solver.read_maze("mazes/"+filename)

    #get all directories in /results/mazes/filename

    #for each directory, get file with name res.txt

    #for each file, read the results and store them in a list

    dirs= os.listdir("results/mazes/"+filename)

    #read all the results from the files
    results = []

    for dir in dirs:
        with open("results/mazes/"+filename+"/"+dir+"/res.txt") as f:

            
            h={
                "fitness":float(f.readline()),
                "time_to_best":int(f.readline()),
                "solution":[int(x) for x in f.readline().strip()],
                "params":eval(f.readline()),
                "gen":eval(f.readline())

            }
            results.append(h)


    
    #sort the results by fitness
    results.sort(key=lambda x: x["fitness"])

    #get the best result
    best = results[0]

    #figure 
    fig = plt.figure()

    #find min and max of all gen of all results
    min_gen = min([min(x["gen"]) for x in results])
    max_gen = max([max(x["gen"]) for x in results])

    #plot all the results on that scale




    #plot gen of all results
    for r in results:
        plt.plot(r["gen"])

    #add scale
    plt.axis([0,len(best["gen"]),min_gen,max_gen])

    #add title
    plt.title("Fitness: "+str(best["fitness"]))

    #show a dot at time_to_best of best result
    plt.plot(best["time_to_best"],best["gen"][best["time_to_best"]],"ro")

    #show the plot
    plt.show()


    simulate_chromosomes(maze,maze_start, [best["solution"]],title="", auto=True, draw=True)

    


    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        main("maze_treasure_4.txt")
    else:  
        main(sys.argv[1])

