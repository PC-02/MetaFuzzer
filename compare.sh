#!/bin/bash

tmux new-session -d

tmux split-window	# Vertical Split
tmux split-window -h	# Horizontal Split

tmux select-pane -t 0	# Switch to Pane 1
tmux split-window -h	# Horizontal Split

# Run Fuzzer
tmux select-pane -t 0
tmux send 'printf "****** \n\n Neural Network Fuzzer \n\n******\n"' ENTER;
tmux send './fuzzer -i size_in/ -o nn_size_seeds/ -p 12001 -l 10000 ./size @@' ENTER;

# Run NN Module
tmux select-pane -t 1
tmux send 'printf "****** \n\n Neural Network Module \n\n******\n"' ENTER;
tmux send 'python3 module.py ./nn_size_seeds/ ./size NN False False 12001' ENTER;

# Run Fuzzer
tmux select-pane -t 2
tmux send 'printf "****** \n\n LSTM Fuzzer \n\n******\n"' ENTER;
tmux send './fuzzer -i size_in/ -o lstm_size_seeds/ -p 12002 -l 10000 ./size @@' ENTER;

# Run LSTM Module
tmux select-pane -t 3
tmux send 'printf "****** \n\n LSTM Module \n\n******\n"' ENTER;
tmux send 'python3 module.py ./lstm_size_seeds/ ./size LSTM False False 12002' ENTER;


tmux select-pane -t 0
tmux -2 attach-session -d

