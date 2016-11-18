# Train the BNN

Training should be done off the embedded device. This directory provides a framework for defining and training BNNs in TensorFlow, then saving the weights so that they can be ported to an embedded device.

Model creators for Binary NN.

Reference paper: [https://arxiv.org/pdf/1602.02830v3.pdf](https://arxiv.org/pdf/1602.02830v3.pdf)

The script itself trains a binary network to perform a simple XOR classification task (although this could be extended to other tasks). The weights are saved to an output file which can be read by the C evaluation code (in particular, eval/run.c demonstrates how to build input vectors and run them through the network).

From start to finish, models can be trained and evaluated using:

    make eval/run
    python train/model.py --save_path models/model.def
    cat models/model.def | eval/run
