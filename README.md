# ConvCRFs-training
The code is a realization of training code of ConvCRFs

The file convcrf.py is modified from [https://github.com/MarvinTeichmann/ConvCRF](https://github.com/MarvinTeichmann/ConvCRF), and part of training codes are modified from the kaggle notebook [https://www.kaggle.com/code/jokerak/fcn-voc](https://www.kaggle.com/code/jokerak/fcn-voc)

## Realization

I tried to train unary (fcn-resent101) using the same config as the paper, and here is the result

|  Unary   | epochs |  Global ACC |  mIoU   |
| --- | -------- | -------- | --- |
| paper    |  200 |    91.84      |  71.23   |
| ours    | 25+ft | 93.33    |  71.00    |

Besides, I also tried to add some feature vectors into message passing function (+C means choose conv1x1 as the compatibility transformation and +F means add unary output as the feature vector)

|  CRFs   | method |  Global ACC |  mIoU   |
| --- | -------- | -------- | --- |
| paper    | +C (11) |    94.01      |  72.30   |
| ours    | +C (7) | 93.51    |  71.85    |
| ours    | +C + F | 93.52    |  71.92    |
