#open a file of mazes and read mazes seperatd by a blank line

import sys

if __name__ == "__main__":

    
    maze=""
    cnt=0
    with open("mazes_harder.txt","r") as f:
        print("dela")
        for line in f:
            print("dela2")

            #check if line is empty
            if line.strip():
                print(maze)
                
                #write to file 

                with open("mazes/maze_hard"+str(cnt)+".txt","w") as f2:
                    f2.write(maze)
                    f2.close()
                maze=""
                cnt+=1
            else:
                maze+=line

                
        