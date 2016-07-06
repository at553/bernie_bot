# bernie_bot

Generate artificial reddit posts about Bernie Sanders.

## How does this work?

Uses a high-level api built on top of Tensorflow to define and train a RNN on a corpus of reddit posts scraped from 
[r/sandersforpresident](reddit.com/r/sandersforpresident). This network learns character sequences and uses them to generate novel posts like those found on reddit.


### Notes

Recommended for use only on systems with substantial graphics processing power -- on a MacBook Pro with integrated graphics, this will take somewhere in the order of weeks to train completely

Inspired by the [tflearn shakespeare tutorial](https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm_generator_shakespeare.py)



