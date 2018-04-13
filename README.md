
# LM-IT-DIL-CNN-CRF
A Hybrid Experiment Combining LM_LSTM_CRF model with Iterated Dilated Convolutions Model for Named Entity Recognition <br/>

This Repository heavily borrows code from LM_LSTM_CRF official implementation https://github.com/LiyuanLucasLiu/LM-LSTM-CRF/ and Tensorflow code for Iterated Dilated Convolutions is translated to Pytorch Implementation https://github.com/iesl/dilated-cnn-ner <br/>

**TO-DO**
I felt the loss function used in Iterated Dilated Convolutions are complex and need to understand them well since it has really good regularization terms in overall.<br/>

Need to Understand the CRF completely, I just used the CRF layer provided by LM_LSTM_CRF <br/>

Need to check with multiple hyper parameters for Iterated-Dilated-Convolutions <br/>

Since This is still an experimental Repo, the files are unclear for a beginner, but if you are eager you can explore the lm_lstm_crf.py file in models/ for implementation of iterated Dilated Convolutions in pytorch. <br/>
Happy Hacking <br/>

Details regarding the experiments will be updated once experiments are done. <br/>
**Training**
Training is similar to LM_LSTM_CRF except the extra hyper parameters for iterated dilated convolutions need to changed at the lm_lstm_crf file in the models folder https://github.com/nikhilrayaprolu/LM-IT-DIL-CNN-CRF/blob/master/model/lm_lstm_crf.py#L40


## Installation

For training, a GPU is strongly recommended for speed. CPU is supported but training could be extremely slow.

### PyTorch

The code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/). 

### Dependencies

The code is written in Python 3.6. Its dependencies are summarized in the file ```requirements.txt```. You can install these dependencies like this:
```
pip3 install -r requirements.txt
```

## Data

We mainly focus on the CoNLL 2003 NER dataset, and the code takes its original format as input. 
However, due to the license issue, we are restricted to distribute this dataset.
You should be able to get it [here](http://aclweb.org/anthology/W03-0419).
You may also want to search online (e.g., Github), someone might release it accidentally.

### Format

We assume the corpus is formatted as same as the CoNLL 2003 NER dataset.
More specifically, **empty lines** are used as separators between sentences, and the separator between documents is a special line as below.
```
-DOCSTART- -X- -X- -X- O
```
Other lines contains words, labels and other fields. **Word** must be the **first** field, **label** mush be the **last**, and these fields are **separated by space**.
For example, the first several lines in the WSJ portion of the PTB POS tagging corpus should be like the following snippet.

```
-DOCSTART- -X- -X- -X- O

Pierre NNP
Vinken NNP
, ,
61 CD
years NNS
old JJ
, ,
will MD
join VB
the DT
board NN
as IN
a DT
nonexecutive JJ
director NN
Nov. NNP
29 CD
. .


```

## Usage

Here we provide implementations for two models, one is **LM-LSTM-CRF** and the other is its variant, **LSTM-CRF**, which only contains the word-level structure and CRF.
```train_wc.py``` and ```eval_wc.py``` are scripts for LM-LSTM-CRF, while ```train_w.py``` and ```eval_w.py``` are scripts for LSTM-CRF.
The usages of these scripts can be accessed by the parameter ````-h````, i.e., 
```
python train_wc.py -h
python train_w.py -h
python eval_wc.py -h
python eval_w.py -h
```

The default running commands for NER and POS tagging, and NP Chunking are:

- Named Entity Recognition (NER):
```
python train_wc.py --train_file ./data/ner/train.txt --dev_file ./data/ner/testa.txt --test_file ./data/ner/testb.txt --checkpoint ./checkpoint/ner_ --caseless --fine_tune --high_way --co_train --least_iters 100
```

- Part-of-Speech (POS) Tagging:
```
python train_wc.py --train_file ./data/pos/train.txt --dev_file ./data/pos/testa.txt --test_file ./data/pos/testb.txt --eva_matrix a --checkpoint ./checkpoint/pos_ --caseless --fine_tune --high_way --co_train
```

- Noun Phrase (NP) Chunking:
```
python train_wc.py --train_file ./data/np/train.txt.iobes --dev_file ./data/np/testa.txt.iobes --test_file ./data/np/testb.txt.iobes --checkpoint ./checkpoint/np_ --caseless --fine_tune --high_way --co_train --least_iters 100
```

For other datasets or tasks, you may wanna try different stopping parameters, especially, for smaller dataset, you may want to set ```least_iters``` to a larger value; and for some tasks, if the speed of loss decreasing is too slow, you may want to increase ```lr```.

## Benchmarks

Here we compare LM-LSTM-CRF with recent state-of-the-art models on the CoNLL 2000 Chunking dataset, the CoNLL 2003 NER dataset, and the WSJ portion of the PTB POS Tagging dataset. All experiments are conducted on a GTX 1080 GPU.

A serious bug was found on the ```bioes_to_span``` function, we are doing experiments and would update the results of NER & Chunking later.

### NER

When models are only trained on the WSJ portion of the PTB POS Tagging dataset, the results are summarized as below.

|Model | Max(Acc) | Mean(Acc) | Std(Acc) | Time(h) |
| ------------- |-------------| -----| -----| ---- |
| LM-LSTM-CRF | **91.35** | **91.24** | 0.12 | 4 |
| -- HighWay | 90.87 | 90.79 | 0.07 | 4 |
| -- Co-Train | 91.23 | 90.95 | 0.34 | 2 |

### POS

When models are only trained on the WSJ portion of the PTB POS Tagging dataset, the results are summarized as below.

|Model | Max(Acc) | Mean(Acc) | Std(Acc) | Reported(Acc) | Time(h) |
| ------------- |-------------| -----| -----| -----| ---- |
| [Lample et al. 2016](https://github.com/glample/tagger) | 97.51 | 97.35 | 0.09 | N/A | 37 |
| [Ma et al. 2016](https://github.com/XuezheMax/LasagneNLP) | 97.46 | 97.42 | 0.04 | 97.55 | 21 |
| LM-LSTM-CRF | **97.59** | **97.53** | 0.03 | | 16 |

## Pretrained Model

### Evaluation

We released pre-trained models on these three tasks. The checkpoint file can be downloaded at the following links. Notice that the NER model and Chunking model (coming soon) are trained on both the training set and the development set:

| WSJ-PTB POS Tagging |  CoNLL03 NER |
| ------------------- |
| [Args](https://drive.google.com/a/illinois.edu/file/d/0B587SdKqutQmN1UwNjhHQkhUWEk/view?usp=sharing) | [Args](https://drive.google.com/file/d/1tGAQ0hu9AsIBdrqFn5fmDQ72Pk1I-o74/view?usp=sharing) | 
| [Model](https://drive.google.com/a/illinois.edu/file/d/0B587SdKqutQmSDlJRGRNandhMGs/view?usp=sharing) | [Model](https://drive.google.com/file/d/1o9kjZV5EcHAhys3GPgl7EPGE5fuXyYjr/view?usp=sharing) | 

Also, ```eval_wc.py``` is provided to load and run these checkpoints. Its usage can be accessed by command ````python eval_wc.py -h````, and a running command example is provided below:
```
python eval_wc.py --load_arg checkpoint/ner/ner_4_cwlm_lstm_crf.json --load_check_point checkpoint/ner_ner_4_cwlm_lstm_crf.model --gpu 0 --dev_file ./data/ner/testa.txt --test_file ./data/ner/testb.txt
```

### Prediction

To annotated raw text, ```seq_wc.py``` is provided to annotate un-annotated text. Its usage can be accessed by command ````python seq_wc.py -h````, and a running command example is provided below:
```
python seq_wc.py --load_arg checkpoint/ner/ner_4_cwlm_lstm_crf.json --load_check_point checkpoint/ner_ner_4_cwlm_lstm_crf.model --gpu 0 --input_file ./data/ner2003/test.txt --output_file output.txt
```

The input format is similar to CoNLL, but each line is required to only contain one field, token. For example, an input file could be:

```
-DOCSTART-

But
China
saw
their
luck
desert
them
in
the
second
match
of
the
group
,
crashing
to
a
surprise
2-0
defeat
to
newcomers
Uzbekistan
.
```
and the corresponding output is:

```
-DOCSTART- -DOCSTART- -DOCSTART-

But <LOC> China </LOC> saw their luck desert them in the second match of the group , crashing to a surprise 2-0 defeat to newcomers <LOC> Uzbekistan </LOC> . 

```

## Reference

```
@inproceedings{2017arXiv170904109L,
  title = "{Empower Sequence Labeling with Task-Aware Neural Language Model}", 
  author = {{Liu}, L. and {Shang}, J. and {Xu}, F. and {Ren}, X. and {Gui}, H. and {Peng}, J. and {Han}, J.}, 
  booktitle={AAAI},
  year = 2018, 
}
```
