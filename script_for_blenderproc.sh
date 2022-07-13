#!/bin/bash

FRONT=$1
FUTURE=$2
PANOPTIC=$3
for FILE in $(ls $FRONT)
do
    SCENE=${FILE::-5} # name without .json
    if [ -d $PANOPTIC/$SCENE/additional ] 
    then
        continue # additional folder already exists
    fi
    python run.py own/front_3d/config_additional.yaml $FRONT/$FILE $FUTURE $PANOPTIC/$SCENE $PANOPTIC/$SCENE/additional || break
    trap "echo Exited!; exit;" SIGINT SIGTERM
    echo $SCENE >> additional-scenes-created.txt
done
