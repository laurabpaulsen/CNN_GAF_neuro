source env/bin/activate


# preprocess EEG data
echo "[INFO]: Preprocessing EEG data"
python src/preprocess_eeg.py

echo "[INFO]: Getting GAFs and MTFs for all subjects"
python src/timeseries2gaf.py

# print info
echo "[INFO]: Running CNNs for all subjects"

# run all subjects
subjects = "sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-15 sub-16"
for subject in $subjects
echo "[INFO]: Running CNN for" $subject
python src/cnntorch.py --sub $subject
done