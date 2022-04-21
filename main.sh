#!/bin/bash

dir="$2_$1_seeds"

# Check if program file exists
if [ ! -f "$1" ]
then
        printf "$1 does not exist\n"
        exit 1
fi


# Check if seed input directory exists
if [ ! -d "$1_in" ]
then
        printf "$1 input seed directory does not exist\n"
        exit 1
fi


# Pick model to run
if [ "$2" = "NN" ]
then
        type="Neural Network"
elif [ "$2" = "LSTM" ]
then
        type="LSTM"
else
        printf "Please pick either NN/LSTM as the model\n"
        exit 1
fi

# make output directory if it does not exist
if [ ! -d $dir ]
then
	mkdir $dir
fi

tmux new-session -d
tmux split-window -h

# Start fuzzer
tmux select-pane -t 0
tmux send "printf '****** \n\n $type Fuzzer \n\n******\n'" ENTER;
tmux send "./fuzzer -i $1_in/ -o $dir/ -p 12000 -l 10000 ./$1 @@" ENTER;

# Start module
tmux select-pane -t 1
tmux send "printf '****** \n\n $type Module \n\n******\n'" ENTER;
tmux send "python3 module.py $dir/ ./$1 $2 False False 12000" ENTER;

tmux select-pane -t 0
tmux -2 attach-session -d

