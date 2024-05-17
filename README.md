# CS6910_Assignment3
#Bhavik More 

#A Transliteration System employing RNN, GRU, and LSTM, integrated into Encoder-Decoder Seq2Seq Architectures (with and without Attention Mechanisms).

##Assignment Description:
The project focuses on developing a transliteration system using advanced neural network architectures like RNNs, GRUs, and LSTMs. It employs encoder-decoder Seq2Seq models, both with and without attention mechanisms, to convert text between different scripts while preserving phonetic structures. The goal is to compare these models using the Aksharantar dataset from AI4Bharat, aiming to identify the most effective approach for accurate transliteration. Hyperparameter tuning via wandb is used to optimize model performance.

The Complete Step by step procedure to Build and train the model is given in the jupyter notebook in the current directory.

Pytorch was used so as to create the model and WandB was used for hyperparameter tuning.

Hyperparameters were tuned for two variations of the model:
a)Without Attention mechanism:
The search space for hyperparameter tuning is described below:
```
sweep_config = {
    "method": "bayes",
    'metric': {
        'name': 'Val_Accuracy',
        'goal': 'maximize'
    },
    "parameters": {
        "epochs": {"values": [ 5, 10 , 15]},
        "lr": {"values": [1e-3, 1e-4]},
        "cell_type": {"values": ["RNN","LSTM", "GRU"]},
        "bidirectional": {"values": [True, False]},
        "enc_lyr": {"values": [1,2, 3,4]},
        "dec_lyr": {"values": [1,2, 3,4]},
        "batch_size": {"values": [32,64,128]},
        "embedding_dim": {"values": [32,64,128]},
        "hidden_lyr": {"values": [64,128,256]},
        "encoder_dropout": {"values": [0, 0.1, 0.2]},
        "decoder_dropout": {"values": [0, 0.1, 0.2]},
        "attention": {"values": [False]}
    }
}
```



b) With Attention mechanism:
The search space for hyperparameter tuning is described below:
```
sweep_config = {
    "method": "bayes",
    'metric': {
        'name': 'Val_Accuracy',
        'goal': 'maximize'
    },
    "parameters": {
        "epochs": {"values": [ 10]},
        "lr": {"values": [1e-4]},
        "cell_type": {"values": ["RNN","LSTM", "GRU"]},
        "bidirectional": {"values": [True, False]},
        "enc_lyr": {"values": [1,2]},
        "dec_lyr": {"values": [1,2]},
        "batch_size": {"values": [128]},
        "embedding_dim": {"values": [32,64]},
        "hidden_lyr": {"values": [128,256]},
        "encoder_dropout": {"values": [0, 0.1]},
        "decoder_dropout": {"values": [0, 0.1]},
        "attention": {"values": [True]}
    }
}
```
WANDB Report Link:https://wandb.ai/ch22m009/DLA3/reports/Assignment-3--Vmlldzo3OTkxMDY0
