# COHA cooccurence matrix
This is code to generate cooccurence matrix from COHA dataset. The code is based on the code from [this repository](https://github.com/williamleif/histwords), which has been adapted to support Python 3. The code will generate a cooccurence matrix for each decade in the [COHA dataset](https://www.english-corpora.org/coha/). 

## Install dependencies
```
pip install -r requirements.txt
```

## Data structure
Data should be stored in the following structure:

    data/
    ├── 1810/
    ├── 1820/
    ├── 1830/
    ├── 1840/
    ├── ...
    |__ 2010/

Where each folder contains the data fot a given decade in the raw format of the COHA dataset. That is, each like has a word/lemma/PoS (not continuous text).

## Usage
Usage assumes that the data is structured as above.

* First, we have to build an index for the word/lemma/PoS files:

        $ python src/buildindexes.py
    
    Run `python src/buildindexes.py --help` for more information. 

* Then, we can generate the cooccurence matrix for a given decade:

        $ python src/makecooccur.py

    Output should be saved in `cooccurs/{type}/{window_size:d}/` where `type` is one of `word`, `lemma`, or `pos` and window size is the size of the window used to generate the cooccurence matrix.

    Run `python src/makecooccur.py  --help` for more information. 

* To read a `.bin` file created by `makecooccur.py` run the following:

        $ python src/example.py --bin-file=<path/to/bin>

    Run `python src/example.py --help` for more information. 

## Notes
This requires `cython` to use `pyximport` to load `src/sparse_io.pyx` from other scripts in `src/`.