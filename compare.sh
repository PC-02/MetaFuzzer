#!/bin/bash

lstm_dir="NN_$1_seeds"
nn_dir="LSTM_$1_seeds"

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


# make output directory if it does not exist
if [ ! -d $nn_dir ]
then
        mkdir $nn_dir
fi


# make output directory if it does not exist
if [ ! -d $lstm_dir ]
then
        mkdir $lstm_dir
fi


tmux new-session -d

tmux split-window	# Vertical Split
tmux split-window -h	# Horizontal Split

tmux select-pane -t 0	# Switch to Pane 1
tmux split-window -h	# Horizontal Split

# Run Fuzzer
tmux select-pane -t 0
tmux send "printf '****** \n\n Neural Network Fuzzer \n\n******\n'" ENTER;
tmux send "./fuzzer -i $1_in/ -o $nn_dir/ -p 12001 -l 10000 $1 @@" ENTER;

# Run NN Module
tmux select-pane -t 1
tmux send "printf '****** \n\n Neural Network Module \n\n******\n'" ENTER;
tmux send "python3 module.py $nn_dir/ $1 NN False False 12001" ENTER;

# Run Fuzzer
tmux select-pane -t 2
tmux send "printf '****** \n\n LSTM Fuzzer \n\n******\n'" ENTER;
tmux send "./fuzzer -i $1_in/ -o $lstm_dir/ -p 12002 -l 10000 $1 @@" ENTER;

# Run LSTM Module
tmux select-pane -t 3
tmux send "printf '****** \n\n LSTM Module \n\n******\n'" ENTER;
tmux send "python3 module.py $lstm_dir/ $1 LSTM False False 12002" ENTER;


tmux select-pane -t 0
tmux -2 attach-session -d

