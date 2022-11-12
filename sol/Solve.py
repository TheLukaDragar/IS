#create a class to solve the problem


class Solve:

    #initialize the class
    def __init__(self, filename,params):
        print("running ", filename)
        self.filename = filename
        self.params = params
        import is2 as is2
        self.is2=is2


    #run
    def run(self):
        #call is2 to solve the maze
        maze,maze_start,maze_end,solution, solution_fitness, solution_idx = self.is2.start_ga(self.params)

        #get id of is2 python process
        import os
        pid = os.getpid()

        print("finished ", self.filename, " pid: ", pid)
        

        return self.filename, maze,maze_start,maze_end,solution, solution_fitness, solution_idx


    