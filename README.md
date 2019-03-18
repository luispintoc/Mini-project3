# Mini-project3
In order to run these scripts, the data files should be in the same folder. Also, the packages used are torch, torchvision, numpy, matplotlib and pandas.

**cnn_3layers.py**
* Architecture: (Conv layer(K=5, S=1, P=0) -> ReLU -> 2x2 MaxPooling -> Dropout)x3 -> (Linear activ. -> dropout)x2)
* Best validation accuracy: 87%

**best_model.py**
* Architecture: (Conv layer(K=4, S=1, P=1) -> ReLU -> 2x2 MaxPooling -> Dropout)x2 -> (Conv layer(K=3, S=1, P=1) -> ReLU -> 2x2 MaxPooling -> Dropout)x2 -> (Linear activ. -> dropout)x2)
* Best validation accuracy: 92.7%
* Best test accuracy: 93.4%

**script_to_submit.py**
* *Task: Uses the best_model.py architecture to output a csv file to submit on the Kaggle competition*
* Uses the CSV package
