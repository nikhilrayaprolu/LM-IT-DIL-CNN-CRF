# LM-IT-DIL-CNN-CRF
A Hybrid Experiment Combining LM_LSTM_CRF model with Iterated Dilated Convolutions Model for Named Entity Recognition <br/>

This Repository heavily borrows code from LM_LSTM_CRF official implementation https://github.com/LiyuanLucasLiu/LM-LSTM-CRF/ and Tensorflow code for Iterated Dilated Convolutions is translated to Pytorch Implementation https://github.com/iesl/dilated-cnn-ner <br/>

**TO-DO**
I felt the loss function used in Iterated Dilated Convolutions are complex and need to understand them well since it has really good regularization terms in overall.<br/>

Need to Understand the CRF completely, I just used the CRF layer provided by LM_LSTM_CRF <br/>

Need to check with multiple hyper parameters for Iterated-Dilated-Convolutions
