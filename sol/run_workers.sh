echo "Spawning 20 workers"

_term() {
    #get all deamons and kill them
    echo "Caught SIGTERM signal!"
    kill -TERM $(jobs -p)
    exit 0


}

#catch crt+c or any other signal

trap _term SIGTERM

trap _term SIGINT

trap _term SIGQUIT

trap _term SIGKILL

trap _term SIGSTOP

childern=""


#define int
i=0

#go trough all int from 0 to including 20

#define string
filename="maze_treasure_4.txt"

while [ $i -le 19 ]
do


    #run the python script with anaconda python
    #/Users/carbs/miniforge3/envs/pytorch_m1/bin/python /Users/carbs/Desktop/IS/sol/test.py

    #create deamons and run the python script and redirect the output to dev/null 
    #/Users/carbs/miniforge3/envs/pytorch_m1/bin/python /Users/carbs/Desktop/IS/sol/is3.py mazes/$file &
    #/Users/carbs/miniforge3/envs/ai/bin/python /Users/carbs/Desktop/IS/sol/is3.py mazes/$file &
    /Users/carbs/miniforge3/envs/ai/bin/python /Users/carbs/Desktop/IS/sol/test2.py -filename $filename -worker $i & 

    child=$!
    #echo "Spawned deamon with PID $child"
    childern="$childern $child"

    #increment int
    i=$((i+1))

    #delay for 0.2 second
    sleep 0.2

    


done

#scan childern and wait for them to die
for child in $childern; do
    while kill -0 "$child" 2>/dev/null; do
        sleep 1
    done
done
