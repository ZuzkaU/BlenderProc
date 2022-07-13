#!/bin/bash

FRONT=$1
FUTURE=$2
PANOPTIC=$3
for FILE in $(ls $FRONT | tail -n +$(cat viewlists-created.txt | wc -l))
do
    SCENE=${FILE::-5} # name without .json
    if ls $PANOPTIC$SCENE/viewlist_* 1> /dev/null 2>&1 # ls returns non-zero if file doesnt exist
    then
        echo $SCENE exist
    elif ls $PANOPTIC$SCENE/campose_* 1> /dev/null 2>&1
    then
        /opt/conda/bin/python run.py own/front_3d/config_viewlist.yaml $FRONT$FILE $FUTURE $PANOPTIC$SCENE output || break
        echo $SCENE >> viewlists-created.txt
    else
        echo $SCENE empty
    fi
    trap "echo Exited!; exit;" SIGINT SIGTERM
done
