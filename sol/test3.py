import pygad

class test3:
    def __init__(self):
        self.h = 1


    def fit(self):
        def fitness(solution, solution_idx):
   
            return 1

        return fitness

    def do(self,name):
        hain = pygad.GA(num_generations=5, num_parents_mating=5, fitness_func=self.fit(),sol_per_pop=10, num_genes=10, init_range_low=0, init_range_high=10, mutation_percent_genes=10)

        hain.run()

        if name == "test5":
            print("i am test5 i change h")
            self.h += 1


        print(name,"h", self.h)