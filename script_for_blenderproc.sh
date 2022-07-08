#!/bin/bash

# 
for FIELD in $(ls ../../front3d-sample)
do
    FILE=$(echo $FIELD".json")
    python run.py own/front_3d/config_additional.yaml ../../3D-FRONT/$FILE ../../3D-FUTURE-model ../../front3d-sample/$FIELD ../../front3d-sample/$FIELD/output
    #python rename_files.py  ../../front3d-sample/$FIELD output
done
