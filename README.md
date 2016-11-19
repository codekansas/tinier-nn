# tinier-nn

A tinier framework for deploying binarized neural networks on embedded systems.

## About

The core of this framework is the use of the Binarized Neural Network (BNN) described in [Binarized Neural Networks:
Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830).
This framework seemed ideal for use with embedded systems such as an Arduino (or Raspberry Pi), but to my knowledge
this wasn't already available.

The system consists of two parts:
  - `train`: TensorFlow code for building a BNN and saving it in a specific framework.
  - `eval`: The inference part, which runs on the system, is written in straight C. It reads the model into SRAM and
    performs matrix multiplications using a bitwise XOR, which (probably) leads to a big improvement in time and power
    consumption (although I haven't benchmarked anything).

The two sample scripts, `train/model.py` and `eval/run.c` demonstrate how to train a model to discriminate an XOR function. The model uses a lot more weights than would theoretically be necessary for this task, but together they demonstrate how to adapt the code to other use cases.

## Demo

To run the demo, run:

    make eval/run
    cat models/model.def | eval/run

The outputs show the predictions for an XOR function.

To train the model, run:

    python train/model.py --save_path models/model.def

This is how the `models/model.def` file was generated.

## Math Notes

Encoding weights / activations with values of -1 and 1 as binary values: `-1 -> 0, 1 -> 1`. Then matrix multiplication
done using the XOR operation. Here's an example:

Using binary weights and activations of -1 and 1:

  - Vector-matrix operation is `[1, -1] * [1, -1; -1, 1] = [1 * 1 + -1 * -1, 1 * -1 + -1 * 1] = [2, -2]`
  - After applying the binary activation function `x > 0 ? 1 : -1` gives `[1, -1]`

Using binary weights and activations of -1 and 1:

  - Encoding the inputs as binary weights: `[1, 0] * [1, 0; 0, 1]`
  - Applying XOR + sum: `[1 ^ 1 + 0 ^ 0; 1 ^ 0 + 0 ^ 1] = [0, 2]`
  - Activation function then becomes `x < (2 / 2) ? 1 : 0` which gives `[1, 0]`

Because the operations are done this way, I made it so that matrix dimensions must be multiples of the integer sizes.
Padding can be used to make data line up correctly (although if someone wants to change this, LMK).

## To Do

  - On most Arduinos, Flash memory is bigger than SRAM by about a factor of 32. So it's not too bad to encode models
    as characters instead of bits (and it makes them easier to debug). Although this is something that could be improved.
  - More examples would be awesome.
  - Matrix multiplication could be better, maybe.
  - Architectures besides feed-forward networks would be good.

<sub>This was a project that I worked on for CalHacks 3.0 (although I never submitted it).</sub>
