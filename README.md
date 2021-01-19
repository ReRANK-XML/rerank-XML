# README for NIPS-2020 code

## Depedencies via Conda Environment
    > conda env create -n xml --file environment.yml
    > source activate xml
**Notice**: the following examples are executed under the > (xml) conda virtual environment

## Install pyxclib
Tools for multi-label classification problems.

    > git clone https://github.com/kunaldahiya/xclib.git
    > cd xclib
    > python setup.py install --user

## Compile Cython Code
We implement some functional interface in Python for the sake of efficiency.

    > python setup.py build_ext --inplace


## Run the Sample
```
#### train models on eurlex using entire dataset
python main.py -d eurlex -ft 1
```

```
#### train models on eurlex by setting label splitting threshold \tau = 0.1
python main.py -d eurlex -thr 0.1
```

```
#### train models on eurlex by setting label splitting threshold \tau = 0.1 and data augmentation parameter n_aug = 3
python main.py -d eurlex -thr 0.1 -aug 3
```