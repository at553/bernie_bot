import tflearn


filename = 'posts.txt'
maxlen = 20

X, Y, char_idx = \
    tflearn.data_utils.textfile_to_semi_redundant_sequences(filename, seq_maxlen=maxlen, redun_step=3)

layer1 = tflearn.input_data([None, maxlen, len(char_idx)])
layer2 = tflearn.lstm(layer1, 256, return_seq=True)

layer3 = tflearn.dropout(layer2, 0.5)
layer4 = tflearn.lstm(layer3, 256, return_seq=True)
layer5 = tflearn.dropout(layer4, 0.5)

layer6 = tflearn.lstm(layer5, 256)
layer7 = tflearn.dropout(layer6, 0.5)

layer8 = tflearn.fully_connected(layer7, len(char_idx), activation='softmax')
layer9 = tflearn.regression(layer8, optimizer='adam', loss='categorical_crossentropy')

seqGen = tflearn.SequenceGenerator(layer9, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0, tensorboard_verbose=True)

for iter in range(75):
    seqGen.fit(X, Y, batch_size=128,
          n_epoch=1, run_id='bernie_bot')


seed = tflearn.data_utils.random_sequence_from_textfile(filename, maxlen)
print(seqGen.generate(250, temperature=1.0, seq_seed=seed))



