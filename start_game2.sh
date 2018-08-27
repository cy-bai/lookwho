#!/bin/bash

game=$1
split=0
while [ $split -lt 9 ]
do
    echo $game $split
    #python game_split_CoNN.py $game $split &
    sleep 1
    (( split++ ))
done
