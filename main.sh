#!/bin/sh

while [ 1 ]
do
    CURRENT_DATE=$(date +"%Y-%m-%d-%T")
    echo "PROCESSING ON ${CURRENT_DATE}"
    ./download.sh ${CURRENT_DATE} && ./train.sh ${CURRENT_DATE} &
    sleep 5
done