# Deep learning based generation of DSC MRI parameter maps using DCE MRI data

PyTorch Official Implementation of the Paper 'Deep Learning-Based Generation of DSC MRI Parameter Maps Using DCE MRI Data'

## Documentation
### Dependencies and Installation
Install dependencies with
```shell
  pip install -r requirements.txt
```

### Prepare the data

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
