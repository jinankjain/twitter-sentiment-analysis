## on CNN
1. current CNN flattens out around 85%. Make the model bigger, add more layers, see swisscheese
2. Validation accuracy same as accuracy here, so validate on 5-10%
3. Glove embedding + word2vec embeddings separate channels of input. Current embedding is being learned.

## on LSTM
1. Try bidirectional/cnn-lstm-... text examples from keras resources
2. char-rnn/lstm models

## on OpenAI sentiment neuron
1. Get features + fc layer classifier on full

## Make ensemble
1. Majority vote on final
2. combine probabilities??
3. Combine features??
