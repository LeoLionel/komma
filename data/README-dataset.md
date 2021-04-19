To compile the training data with the scripts in this directory do the following:

1. Download the Leipzig Corpora Collection German 'web-wrt' corpus with 10M sentences (2.3 GB), unzip.
   Link: https://wortschatz.uni-leipzig.de/en/download/German
2. run `python3 build_corpus.py`
3. run `python3 corpus_statistics.py`
4. install the `fasttext` python package
5. Download the pretrained fasttext word embeddings for German in binary format (7.2 GB), unzip.
   Link: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz
6. run `python3 dataset-v2_from_corpus.py`

Each script may take a few minues to run. Now you can load the dataset files with the jupyter notebook and train the model.


