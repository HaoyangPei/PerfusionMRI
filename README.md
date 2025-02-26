# Deep learning based generation of DSC MRI parameter maps using DCE MRI data

PyTorch Official Implementation of the Paper 'Deep Learning-Based Generation of DSC MRI Parameter Maps Using DCE MRI Data'

## Documentation
### Dependencies and Installation
Install dependencies with
```shell
  pip install -r requirements.txt
```

### Imaging protocol
The details of our imaging protocol are available at: [Imaging Protocol](https://github.com/HaoyangPei/PerfusionMRI/tree/main/Imaging%20Protocol)

### Prepare the data
#### 1. Store the Data in H5 Format
Save the data in an HDF5 (`.h5`) file with the following variables:

- **DCE** `(Z × T × W × H)`
- **MTT** `(Z × W × H)`
- **rCBF** `(Z × W × H)`
- **rCBV** `(Z × W × H)`

Note that 
  - `Z`: Number of slices  
  - `T`: Number of frames  
  - `W`: Width  
  - `H`: Height

#### 2. Save Dataset Filenames
Store the filenames of the training, validation, and test datasets in the respective text files:

- `train.txt` → Contains filenames for training data
- `val.txt` → Contains filenames for validation data
- `test.txt` → Contains filenames for test data

Ensure the filenames are listed one per line in each text file.

### Training
Run the training script with
```shell
  Python train.py \
  --data-path 'data_path' \
  --batch 1 \
  --out-map 'output map types (e.g. MTT/CBV/CBF)' \
  --model-name 'model_name' \
  --model-savepath 'model_save_path'
```
