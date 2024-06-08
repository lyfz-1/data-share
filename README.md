# README

This repository contains our `datasets`, `scripts`, and `experimental results`.

The task description for manual labeling, labeled results, the dataset for classification, scripts for the various classification algorithms are included in the directory  `RQ1`.

The directory `RQ2_3` contains manual check results for the stability of the best classifier from RQ1 in the subdirectory `manual_check`. In addition, the comprehensive experimental results of RQ2 and RQ3 in the paper are in the subdirectory `result`. The `metrics.py` script is an implementation of the BLEU_M2, BLEU_M3, BLEU_DC, ROUGE-L, and METEOR metrics. Before running it, you need to change `y_txt` (the referenced message) and `hyp_txt` (the model's prediction) in the header of the file.

The dataset used for our proposed "Diversion" strategy as well as the different classification algorithms we tried are contained in the directory `RQ4`. Experimental results comparing the conventional "Mixed" strategy and our "Diversion" strategy are presented in the subdirectory `result`.

Note that due to the large size of the files of the best classification models in RQ1 and the "Diversion" strategy, we share them on Google Cloud. Given that the explicit and implicit subsets are also too large, we share them on Google Cloud as well. Related link: 

https://drive.google.com/drive/folders/1sjZvzMzN4aMw9jv1zD1vMk8uhkGOgjtn?usp=drive_link

Please refer to the paper for details of the experiments and the resulting data.

