#!/bin/bash 

nohup python3 $1 2>&1 > output/$1_$(date +"%Y-%m-%d_%H-%M-%S") & 

