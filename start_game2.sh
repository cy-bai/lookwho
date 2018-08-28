#!/bin/bash

game=$1
split=0
while [ $split -le 29 ]
do
    echo $game $split
    python game_split_CoNN.py --game_id=$game --split_id=$split --epochs=50 --time_granularity=10 --layer=3 &
    sleep 3
    (( split++ ))
done
