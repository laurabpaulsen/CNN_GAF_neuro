# Classifying object category using gramian angular field and 3D convolutional networks from EEG data
![gaf_example](data/gaf_sub-01_0_0.png)

## Data
The data used for this project was found on openneuro.org. The data consists of EEG data from 16 subjects. Each subject particpated in two experiments, where only the first was used for the current analysis. Participants were presented with a stream of images at 5 Hz. The stimuli consisted of 200 different images, which can be grouped into animate and inanimate stimuli. 

Instructions on how to download the data can be found in the `readme.md` in the `data` directory. 

## Usage
1. Download the data following the instructions in the  `readme.md` in the `data` directory. 
2. Create a virtual environment and install the dependencies
```
bash setup.sh
```
3. Preprocess the EEG data
```
python src/preprocess_eeg.py
```
4. Get the gramian angular fields (both GAFD and GAFS) and Markow transitional fields for timeseries EEG data
```
python src/timeseries2gaf.py
```
5. Train and test the convolutional neural network
```
python src/cnn.py
```

## Repository structure
