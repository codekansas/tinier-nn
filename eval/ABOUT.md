# Evaluate trained BNN

In principle, the training and inference steps for the BNN should be separate from one another. The files in this directory provide code for performing low-level inference on embedded devices, given a model that has been trained elsewhere. This handles memory management appropriately.

`tinier-nn/models/model.def` (the sample definition) contains weights trained for an XOR function. This file demonstrates how to initialize a model and perform a classification task. These weights were trained and saved using the train/model.py script.

To run normally, the model needs to be piped to stdin (weight loading can be configured differently, depending on the application). From this directory, run the following:

    make run  # To build the file itself.
    cat ../models/model.def | ./run