#!/bin/sh
tmux new-session \; split-window -h \; select-pane -t 0 \; split-window -v 'htop' \; select-pane -t 2 \;