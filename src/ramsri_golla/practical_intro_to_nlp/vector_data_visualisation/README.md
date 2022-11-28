## Environment
You should create a virtualenv with the required dependencies by running
```
make -C ../../../../ virtualenv calling_dir="$PWD"
```

When a new requirement is needed you should add it to `unpinned_requirements.txt` and run
```
make -C ../../../../ update-requirements-txt calling_dir="$PWD"
```
to ensure reproducibility, all requirements are pinned and matched


How to activate the environment
```
source ./.env/bin/activate
```


## Tutorials
| Index | Title | Link | Completed |
| --------------- | --------------- | --------------- | --------------- |
| 01 | Visualization of Word Embedding Vectors using Gensim and PCA | [Link](https://towardsdatascience.com/visualization-of-word-embedding-vectors-using-gensim-and-pca-8f592a5d3354) | :ballot_box_with_check: |
