#!/bin/bash

tmux new-session -d
tmux split-window -h
tmux select-pane -t 0
tmux send './fuzzer -i size_in/ -o nn_size_seeds/ -p 12000 -l 10000 ./size @@' ENTER;
tmux select-pane -t 1
tmux send 'python3 module.py ./nn_size_seeds/ ./size NN False False 12000' ENTER;
tmux select-pane -t 0
tmux -2 attach-session -d

