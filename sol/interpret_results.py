

import sys
import os

from is4 import IS4

from test import *
#import plt

from matplotlib import pyplot as plt




def plotparameter(combination_name,combination_list,results,y="fitness"):

    print("testing ",combination_name)
    print(combination_list)

    #find min
    if y=="fitness":
        m=min([x["fitness"] for x in results])
        print("min: ",m)
        assert m>=0
        

   

        

    fig = plt.figure()
    if y=="fitness":
    #set y axis to max fitness
        max_fitness = max([x["fitness"] for x in results])
        plt.ylim(0,max_fitness)
    #title
    plt.title(combination_name+" vs "+y)
    #the values in combination_list are the values of the parameter we want to test
    

    for value in combination_list:

        avg_fitness = 0
        variance = 0
        for combination in results:
            if combination["params"][combination_name]==value:
                avg_fitness+=combination[y]
                variance+=combination[y]**2

        avg_fitness/=len(results)
        variance/=len(results)
        variance-=avg_fitness**2

        #normalize the variance
        variance/=avg_fitness



        print(combination_name,value)
        print("avg ",y,": ",avg_fitness)
        print("variance: ",variance)

        #plot the results for this population sizes
        #figure
       

        #add average fitness and variance to boxplot at x=population_size
        #calculate the witdh based on the max and min of the combination list
        #check if values are numbers exclude boolean values
        if isinstance(combination_list[0],int) or isinstance(combination_list[0],float):
            width = (max(combination_list)-min(combination_list))/len(combination_list)
            #if value is float round to 2 decimals
            if isinstance(value,float):
                value = round(value,2)
            
            plt.boxplot([avg_fitness,variance],positions=[value],widths=width)

        else:
            #make x a string so it can be used as a label
            plt.boxplot([avg_fitness,variance],positions=[combination_list.index(value)],widths=0.5)
            #x values are strings so we need to set the xticks to the values of the combination list
            #we want to show them
            plt.xticks(range(len(combination_list)),combination_list)

            
            
            
        #make it wide 


    #show the plot

    #on the plot show the best y value





    plt.show()




def main(filename):

    solver = IS4()
    maze,maze_start,maze_end = solver.read_maze("mazes/"+filename)

    #get all directories in /results/mazes/filename

    #for each directory, get file with name res.txt

    #for each file, read the results and store them in a list

    dirs= os.listdir("results/mazes/"+filename)
    #remove .DS_Store
    dirs.remove('.DS_Store')

    #read all the results from the files
    results = []

    for dir in dirs:
        print(dir)
        with open("results/mazes/"+filename+"/"+dir+"/res.txt") as f:

            
            h={
                "fitness":float(f.readline()),
                "time_to_best":int(f.readline()),
                "solution":[int(x) for x in f.readline().strip()],
                "params":eval(f.readline()),
                "gen":eval(f.readline()),
                "worker":dir.split("_")[-1]

            }
            results.append(h)






    parameters = {

        "population_size": range(25, 251, 50),
        
        "population_parents_percent": arange(0.02, 0.20, 0.10),
        "mutation_probability": arange(0.05, 0.20, 0.05),

        "population_func" : ["invalid","valid","valid_smart"],
        "crossover_type":[ "min_max","longest_path_mix","rand_rand"],
        "smart": [1, 0], #smart crossover
        "parent_selection_type": ["sus","rws","random"],


    }

    #plotparameter("population_size",population_sizes,results)
    #plotparameter("population_size",parameters["population_size"],results,y="fitness")

    #plot all the parameters

    #we want to make every fitness non negative
    #so we can plot it

    minnn=min([x["fitness"] for x in results]) 
    print("minnn: ",min([x["fitness"] for x in results]))
    if minnn<0:
        for res in results:
            res["fitness"]+=abs(minnn)


    for parameter in parameters:
        plotparameter(parameter,parameters[parameter],results,y="fitness")

    exit()


    
    #sort the results by fitness
    results.sort(key=lambda x: x["fitness"],reverse=True)

    #get the best result
    best = results[0]

    #figure 
    fig = plt.figure()

    #find min and max of all gen of all results
    min_gen = min([min(x["gen"]) for x in results])
    max_gen = max([max(x["gen"]) for x in results])

    #plot all the results on that scale




    #plot gen of all results
    #for r in results:
        #plt.plot(r["gen"])

    #plot best gen
    plt.plot(best["gen"])

    print("best fitness: ",best["fitness"])
    print("time to best: ",best["time_to_best"])
#worker: ",best["worker"])
    print("worker: ",best["worker"])
    #params
    print("params: ",best["params"])



    #add scale
    plt.axis([0,len(best["gen"]),min_gen,max_gen])

    #add title
    plt.title("Fitness: "+str(best["fitness"]))

    #show a dot at time_to_best of best result
    plt.plot(best["time_to_best"],best["gen"][best["time_to_best"]],"ro")

    #show the plot
    plt.show()

    #we want to analyze all the combinations and how eqach parameter affects the results
    #first lets test population size
    #we want to test the following population sizes: 10,20,30,40,50,60,70,80,90,100
    
    

    #loop through all the population sizes and combinations
    fig = plt.figure()
    for population_size in population_sizes:

        avg_fitness = 0
        variance = 0
        for combination in results:
            if combination["params"]["population_size"]==population_size:
                avg_fitness+=combination["fitness"]
                variance+=combination["fitness"]**2

        avg_fitness/=len(results)
        variance/=len(results)
        variance-=avg_fitness**2

        print("population size: ",population_size)
        print("avg fitness: ",avg_fitness)
        print("variance: ",variance)

        #plot the results for this population sizes
        #figure
       

        #add average fitness and variance to boxplot at x=population_size
        plt.boxplot([avg_fitness,variance],positions=[population_size],widths=10)
        #make it wide 


    #show the plot
    plt.show()


               








    simulate_chromosomes(maze,maze_start, [best["solution"]],title="", auto=True, draw=True)

    


    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        main("maze_treasure_7.txt")
    else:  
        main(sys.argv[1])

