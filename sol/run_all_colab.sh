echo "Spawning 100 processes"

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
#get all files in the maze directory

#do not get files with hard in them

#set env variable for the maze

export PYGAME_HIDE_SUPPORT_PROMPT=1

#define int 
declare -i i=0

for file in $(ls mazes | grep maze_harder_ | grep -E '_0.txt|_1.txt'  ); do
    #run the python script with anaconda python
    #/Users/carbs/miniforge3/envs/pytorch_m1/bin/python /Users/carbs/Desktop/IS/sol/test.py

    #create deamons and run the python script and redirect the output to dev/null 
    /usr/local/envs/ai/bin/python /content/IS/sol/is4.py mazes/$file &
    #/Users/carbs/miniforge3/envs/ai/bin/python /Users/carbs/Desktop/IS/sol/is3.py mazes/$file &
    #/Users/carbs/miniforge3/envs/ai/bin/python /Users/carbs/Desktop/IS/sol/test.py $file &

    #increment the counter

    child=$!
    echo "Spawned deamon with PID $child"
    childern="$childern $child"
    


done

#scan childern and wait for them to die
for child in $childern; do
    while kill -0 "$child" 2>/dev/null; do
        sleep 1
    done
done
