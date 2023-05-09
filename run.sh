#!/bin/bash

# Activate virtual environment
source env/bin/activate

# Preprocess EEG data
echo "[INFO]: Preprocessing EEG data"
# python src/preprocess_eeg.py

# Get GAFs and MTFs for all subjects
echo "[INFO]: Getting features for the CNN for all subjects  (GAFs and MTFs)"
# python src/timeseries2gaf.py

# Define the subjects
subjects="sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-15 sub-16"

# Iterate over each subject
for subject in $subjects; do
    echo "[INFO]: Running CNN for $subject"
    python src/cnntorch.py --sub $subject
done
