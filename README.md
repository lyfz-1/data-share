# README

This repository contains our `datasets`, `scripts`, and `experimental results`.

The task description for manual labeling, labeled results, the dataset for classification, scripts for the various classification algorithms, and the explicit and implicit subsets are included in the directory  `RQ1`. In the `classifiers` subdirectory, we provide the best-performing model file.

The directory `RQ2_3` contains manual check results for the stability of the best classifier from RQ1 in the subdirectory `manual_check`. In addition, the comprehensive experimental results of RQ2 and RQ3 in the paper are in the subdirectory `result`. The `metrics.py` script is an implementation of the BLEU_M2, BLEU_M3, BLEU_DC, ROUGE-L, and METEOR metrics. Before running it, you need to change `y_txt` (the referenced message) and `hyp_txt` (the model's prediction) in the header of the file.

The dataset used for our proposed "Diversion" strategy as well as the models are contained in the directory `RQ4`. Experimental results comparing the conventional "Mixed" strategy and our "Diversion" strategy are presented in the subdirectory `result`.

Please refer to the paper for details of the experiments and the resulting data.