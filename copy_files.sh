#!/bin/bash

# Define the destination directory in Windows
destination_dir="../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN"

# List of files to copy
files=("HESS_CNN_CreateModelsFunctions.py" "HESS_CNN_ProcessingDataFunctions.py" "HESS_CNN_Run.py" "HESS_CNN_ModaMulti_Run.py" "job_HESS_CNN.sh" "job_HESS_CNN_ModaMulti.sh")

# Loop through the files and copy them using scp
for file in "${files[@]}"; do
    scp "$file" "$destination_dir"
done