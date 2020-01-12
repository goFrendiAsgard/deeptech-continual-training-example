#!/bin/sh

mkdir -p data
CURRENT_DATE=$1
if [ -z "${CURRENT_DATE}" ]
then
    CURRENT_DATE=$(date +"%Y-%m-%d-%T")
fi
wget ${GDRIVE_URL} -O data/${CURRENT_DATE}.csv