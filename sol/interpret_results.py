import sys
import os
from is4 import IS4
from test import *
#import plt
from matplotlib import pyplot as plt
import random

def plotparameter(combination_name,combination_list,results,filename,y="fitness",text="",count_only=False):
    print("testing ",text,combination_name)
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

    if y=="time_to_best":
    #set y axis to max fitness
        max_ttb = max([x["time_to_best"] for x in results])
        plt.ylim(0,max_ttb)

    if count_only:
        plt.ylim(0,len(results))

    #title
    plt.title(text+" "+combination_name+" vs "+y)
    #add subtitle
    if y=="fitness":
        plt.suptitle("higher is better")
    if y=="time_to_best":
        plt.suptitle("lower is better")

    if y=="count":
        plt.suptitle("number of results")




    #the values in combination_list are the values of the parameter we want to test
    for value in combination_list:
        count=0
        avg_fitness = 0
        variance = 0
        for combination in results:
            if combination["params"][combination_name]==value:
                count+=1
                avg_fitness+=combination[y]
                variance+=combination[y]**2
        avg_fitness/=len(results)
        variance/=len(results)
        variance-=avg_fitness**2
        #normalize the variance
        if avg_fitness!=0:

            variance/=avg_fitness
        else:
            print("param is no present in results!")
            variance=0

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

            if count_only:
                plt.plot(value,count,"ro")
            else:


                plt.boxplot([avg_fitness,variance],positions=[value],widths=width)

        else:
            #make x a string so it can be used as a label
            if count_only:
                #add point

                plt.bar(value,count,width=0.5)

            else:

                plt.boxplot([avg_fitness,variance],positions=[combination_list.index(value)],widths=0.5)
            #x values are strings so we need to set the xticks to the values of the combination list
            #we want to show them
                plt.xticks(range(len(combination_list)),combination_list)
        #make it wide
    #show the plot
    #on the plot show the best y value
    #save plot to /analysis/filename/combination_name.png

    #check if directory exists
    dir="analysis/"+filename+"/"+combination_name+" vs "+y
    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.savefig(dir+"/plot.png")




    plt.show()

def showbest(results,maze,maze_start,markbest=False,bestbest=None):
    results.sort(key=lambda x: x["fitness"],reverse=True)
    #get the best result
    best = results[0]
    #get the best results may have the same fitness
    best_results_same = [x for x in results if x["fitness"]==best["fitness"]]

    #plot gen of each best result
    fig = plt.figure()
    plt.title("best results: "+ str(len(best_results_same))+" fitness of which is: "+str(best["fitness"]))
    #strt from -1 because we want to start at 0 x axis

    for result in best_results_same:

        if markbest:
            #plot with low alpha
            plt.plot(result["gen"],alpha=0.3)
        else:
            plt.plot(result["gen"])

    #
    if markbest and bestbest:

        #generate random color for bestbest len
        colors = []
        for i in range(len(bestbest)):
            colors.append((random.random(),random.random(),random.random()))
        
        for i , result in enumerate(bestbest):
            plt.plot(result["gen"],color=colors[i],linewidth=1.5)
        
       


    plt.show()



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

    #shuffle results

    #simulate_chromosomes(maze,maze_start,[x["solution"] for x in results if x["fitness"]],title="", auto=True, draw=True)
    simulate_chromosomes(maze,maze_start,[best["solution"]],title="best", auto=True, draw=True)




def main(filename,resultsdir,parameters):

    solver = IS4()
    maze,maze_start,maze_end = solver.read_maze("mazes/"+filename)
    #get all directories in /results/mazes/filename
    #for each directory, get file with name res.txt
    #for each file, read the results and store them in a list
    dirs= os.listdir(resultsdir+"/mazes/"+filename)
    #remove .DS_Store
    dirs = [x for x in dirs if x!=".DS_Store"]



    #check and count all the files
    #must contain filename and end with 0-11
    #count the number of files
    countt=[0]*12
    for d in dirs:
        end=d.split("_")[-1]
        if filename in d and end.isdigit() and int(end)<12:
            print(d)
            countt[int(end)]+=1

    print(countt)
    print("total: ",sum(countt))





    #read all the results from the files
    results = []
    for dir in dirs:
        #print(dir)
        with open(resultsdir+"/mazes/"+filename+"/"+dir+"/res.txt") as f:
            h={
                "fitness":float(f.readline()),
                "time_to_best":int(f.readline()),
                "solution":[int(x) for x in f.readline().strip()],
                "params":eval(f.readline()),
                "gen":eval(f.readline()),
                "worker":dir.split("_")[-1]
            }

            #turn bools to strings
            for key in h["params"]:
                if isinstance(h["params"][key],bool):
                    h["params"][key]=str(h["params"][key])



            results.append(h)


    print("read:" ,len(results),"results")
    #get max population size
    max_population_size = max([x["params"]["population_size"] for x in results])
    print("max population size: ",max_population_size)
    



    showbest(results,maze,maze_start)


    #plotparameter("population_size",population_sizes,results)
    #plotparameter("population_size",parameters["population_size"],results,y="fitness")
    #plot all the parameters
    #so we can plot it
    #sort the results by fitness

    #we want to analyze all the combinations and how eqach parameter affects the results
    #first lets test population size
    #we want to test the following population sizes: 10,20,30,40,50,60,70,80,90,100
    minnn=min([x["fitness"] for x in results])
    print("minnn: ",min([x["fitness"] for x in results]))
    if minnn<0:
        assert False
        for res in results:
            res["fitness"]+=abs(minnn)
    for parameter in parameters:
        try:
            plotparameter(parameter,parameters[parameter],results,filename,y="fitness")


            #plotparameter(parameter,parameters[parameter],results,filename,y="time_to_best")
            pass

        except  Exception as e:
            print("error with parameter: ",parameter)
            print(e)

            continue
    #loop through all the population sizes and combinations

    #get top 30 results based on fitness
    results.sort(key=lambda x: x["fitness"],reverse=True)
    #get ot 10% of the results
    top = results[:int(len(results)*0.1)]

    print("top 10%: ",len(top),"fitness: ",top[0]["fitness"])

    params_can_be_missing=parameters

    #params_can_be_missing["population_size"]

    for parameter in params_can_be_missing:

        #plotparameter(parameter,params_can_be_missing[parameter],top,filename,y="fitness",text="top300")
        plotparameter(parameter,params_can_be_missing[parameter],top,filename,y="fitness",text="top 10%:",count_only=True)

        #plotparameter(parameter,params_can_be_missing[parameter],top,filename,y="time_to_best",text="top300")
        plotparameter(parameter,params_can_be_missing[parameter],top,filename,y="time_to_best",text="top 10%:",count_only=False)



    #get all with best fitness
    best = [x for x in results if x["fitness"]==results[0]["fitness"]]
    print("best: ",len(best),"fitness: ",best[0]["fitness"])

    #params

    for parameter in params_can_be_missing:

        #plotparameter(parameter,params_can_be_missing[parameter],best,filename,y="fitness",text="best")
        plotparameter(parameter,params_can_be_missing[parameter],best,filename,y="fitness",text="best with same fitness:",count_only=True)

        #plotparameter(parameter,params_can_be_missing[parameter],best,filename,y="time_to_best",text="best")
        plotparameter(parameter,params_can_be_missing[parameter],best,filename,y="time_to_best",text="best with same fitness:",count_only=False)



    #find the best result with max fitness and min time to best
    best_best = [] #can be more than one
    for res in best:
        if len(best_best)==0:
            best_best.append(res)
        else:
            #check every result in best_best
            #if the current result is better, clear the list and add the current result
            #if the current result is worse, skip it
            #if the current result is equal, add it to the list
            better = False
            worse = False
            for b in best_best:
                if res["fitness"]>b["fitness"]:
                    worse=True
                elif res["fitness"]<b["fitness"]:
                    better=True
                elif res["fitness"]==b["fitness"]:
                    if res["time_to_best"]<b["time_to_best"]:
                        better=True
                    elif res["time_to_best"]>b["time_to_best"]:
                        worse=True

            if better:
                best_best=[res]
            elif not worse:
                best_best.append(res)
            
    

    print("found best best: ",len(best_best),"fitness: ",best_best[0]["fitness"],"time to best: ",best_best[0]["time_to_best"])


    #plot the best solution gen
    showbest(results,maze,maze_start,markbest=True,bestbest=best_best)

    for best in best_best:
        print("best: ",best["fitness"],"time to best: ",best["time_to_best"],"params: ",best["params"])
    







if __name__ == "__main__":
    if len(sys.argv) != 2:

    #     parameters = {
    #     "population_size": range(25, 251, 50),
    #     "population_parents_percent": arange(0.02, 0.20, 0.10),
    #     "mutation_probability": arange(0.05, 0.20, 0.05),
    #     "population_func" : ["invalid","valid","valid_smart"],
    #     "crossover_type":[ "min_max","longest_path_mix","rand_rand"],
    #     "smart": ["False", "True"], #smart crossover
    #     "parent_selection_type": ["sus","rws","random"],
    # }

        parameters = {
          "smart": ["False", "True"], #smart crossover

        "population_size": [200, 300, 400, 500, 600, 700],

        "population_parents_percent": [0.10,0.20,0.30],

        "population_func" : ["invalid","valid","valid_smart"],
        "crossover_type":[ "min_max","rand_rand"],

        }






        main("maze_treasure_7.txt","results_1",parameters)
    else:
        main(sys.argv[1])
