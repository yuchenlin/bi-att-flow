# Bi-directional Attention Flow for Machine Comprehension
- This is a copy of the original implementation of [Bi-directional Attention Flow for Machine Comprehension][paper] (Seo et al., 2016).
- This is tensorflow v1.4.0 comaptible version. 

## 0. Requirements
#### General
- Python (verified on 3.6.x)
- unzip, wget (for running `download.sh` only)

#### Python Packages
- tensorflow (deep learning library, verified on 1.4.0)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visaulization; if you only train and test, not needed)

## 1. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```

Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
```
python -m squad.prepro
```

## 2. Training
The model has ~2.5M parameters.
The model was trained with NVidia Titan X (Pascal Architecture, 2016).
The model requires at least 12GB of GPU RAM.
If your GPU RAM is smaller than 12GB, you can either decrease batch size (performance might degrade),
or you can use multi GPU (see below).
The training converges at ~18k steps, and it took ~4s per step (i.e. ~20 hours).

Before training, it is recommended to first try the following code to verify everything is okay and memory is sufficient:
```
python -m basic.cli --mode train --noload --debug
```

Then to fully train, run:
```
python -m basic.cli --mode train --noload
```

You can speed up the training process with optimization flags:
```
python -m basic.cli --mode train --noload --len_opt --cluster
```
You can still omit them, but training will be much slower.

Note that during the training, the EM and F1 scores from the occasional evaluation are not the same with the score from official squad evaluation script. 
The printed scores are not official (our scoring scheme is a bit harsher).
To obtain the official number, use the official evaluator (copied in `squad` folder, `squad/evaluate-v1.1.py`). For more information See 3.Test.


## 3. Test
To test, run:
```
python -m basic.cli
```

Similarly to training, you can give the optimization flags to speed up test (5 minutes on dev data):
```
python -m basic.cli --len_opt --cluster
```

This command loads the most recently saved model during training and begins testing on the test data.
After the process ends, it prints F1 and EM scores, and also outputs a json file (`$PWD/out/basic/00/answer/test-####.json`,
where `####` is the step # that the model was saved).
Note that the printed scores are not official (our scoring scheme is a bit harsher).
To obtain the official number, use the official evaluator (copied in `squad` folder) and the output json file:

```
python squad/evaluate-v1.1.py $HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-####.json
```

### 3.1 Loading from pre-trained weights
NOTE: this version is not compatible with the following trained models. 
For compatibility, use [v0.2.1][v0.2.1]. 

Instead of training the model yourself, you can choose to use pre-trained weights that were used for [SQuAD Leaderboard][squad] submission.
Refer to [this worksheet][worksheet] in CodaLab to reproduce the results.
If you are unfamiliar with CodaLab, follow these simple steps (given that you met all prereqs above):

1. Download `save.zip` from the [worksheet][worksheet] and unzip it in the current directory.
2. Copy `glove.6B.100d.txt` from your glove data folder (`$HOME/data/glove/`) to the current directory.
3. To reproduce single model:
  
  ```
  basic/run_single.sh $HOME/data/squad/dev-v1.1.json single.json
  ```
  
  This writes the answers to `single.json` in the current directory. You can then use the official evaluator to obtain EM and F1 scores. If you want to run on GPU (~5 mins), change the value of batch_size flag in the shell file to a higher number (60 for 12GB GPU RAM). 
4. Similarly, to reproduce ensemble method:
  
  ```
  basic/run_ensemble.sh $HOME/data/squad/dev-v1.1.json ensemble.json 
  ```
  If you want to run on GPU, you should run the script sequentially by removing '&' in the forloop, or you will need to specify different GPUs for each run of the for loop.

## Results 