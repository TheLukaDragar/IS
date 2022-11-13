#open a file of mazes and read mazes seperatd by a blank line

import sys

if __name__ == "__main__":

    
    maze=""
    cnt=0
    with open("mazes_harder.txt","r") as f:
        print("dela")
        for line in f:
            

            #mazes are in new lines and seperated by a blank line
            #if line is blank, then we have a new maze and print

            if line.strip() == "":
                print("Maze: ",cnt)
                print(maze)

                #save maze to file
                with open("mazes/maze_harder_"+str(cnt)+".txt","w") as f2:
                    #do not include the last newline
                    f2.write(maze[:-1])

                    f2.close()

                #reset maze


                maze=""
                cnt+=1

            else:
                maze+=line

            



    

                
        