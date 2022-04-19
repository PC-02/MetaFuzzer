#!/bin/bash

tmux new-session -d
tmux split-window -h
tmux select-pane -t 0
tmux send './fuzzer -i size_in/ -o size_seeds/ -l 10000 ./size @@' ENTER;
tmux select-pane -t 1
tmux send 'python3 module.py ./size_seeds/ ./size NN False False' ENTER;
tmux select-pane -t 0
tmux -2 attach-session -d

