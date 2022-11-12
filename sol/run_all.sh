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
for file in $(ls mazes | grep tre | grep -E '5|6|7' ); do
    #run the python script with anaconda python
    #/Users/carbs/miniforge3/envs/pytorch_m1/bin/python /Users/carbs/Desktop/IS/sol/test.py

    #create deamons and run the python script and redirect the output to dev/null 
    /Users/carbs/miniforge3/envs/pytorch_m1/bin/python /Users/carbs/Desktop/IS/sol/is2.py mazes/$file &
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