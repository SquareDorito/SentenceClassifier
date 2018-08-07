'''
Basic implementation of Skip-Thoughts model for calculating sentence embeddings.

Keras implementation by @KenNoh
'''

import recurrentshop
import seq2seq
from recurrentshop.cells import LSTMCell
from seq2seq import LSTMDecoderCell
from seq2seq import LSTMCell
from recurrentshop.engine import RecurrentContainer, RecurrentSequential
from keras.layers import Input, Embedding, Dropout, TimeDistributed, Dense, Activation, Lambda
from keras.layers import add, multiply, concatenate
from keras import backend as K
from keras.models import Model

def build_decoder(dropout, embed_dim, output_length, output_dim, peek=False, unroll=False):
    decoder = RecurrentContainer(
        readout='add' if peek else 'readout_only', output_length=output_length, unroll=unroll, decode=True,
        input_length=shape[1]
    )
    # for i in range(depth[1]):
    decoder.add(Dropout(dropout, batch_input_shape=(None, embed_dim)))
    decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=embed_dim, batch_input_shape=(shape[0], embed_dim)))

    return decoder

def SkipThoughtModel(
        sent_len,
        vocab_size,
        embed_dims,
        output_length,
        output_dim,
        dropout=0.4,
        unroll=False,
        teacher_force=False):
    input_sent = Input(shape=(sent_len, vocab_size), dtype=K.floatx())
    input_sent._keras_history[0].supports_masking = True

    encoder = RecurrentContainer(
        readout=True, input_length=sent_len, unroll=unroll, stateful=False
    )
    # for i in range(depth[0]):
    encoder.add(LSTMCell(embed_dims, batch_input_shape=(None, embed_dims)))
    encoder.add(Dropout(dropout))

    dense1 = TimeDistributed(Dense(embed_dims))
    dense1.supports_masking = True
    dense2 = Dense(embed_dims)

    encoded_seq = dense1(input)
    encoded_seq = encoder(encoded_seq)

    states = [None] * 2
    encoded_seq = dense2(encoded_seq)
    inputs = [input]
    if teacher_force:
        truth_tensor_prev = Input(batch_shape=(None, output_length, output_dim))
        truth_tensor_prev._keras_history[0].supports_masking = True
        truth_tensor_next = Input(batch_shape=(None, output_length, output_dim))
        truth_tensor_next._keras_history[0].supports_masking = True
        inputs += [truth_tensor_prev, truth_tensor_next]

    prev_decoder = build_decoder(dropout=dropout, unroll=unroll, output_length=output_length)
    next_decoder = build_decoder()
    prev_decoded_seq = prev_decoder(
        {'input': encoded_seq, 'ground_truth': inputs[1] if teacher_force else None, 'initial_readout': encoded_seq,
         'states': states})

    next_decoded_seq = next_decoder(
        {'input': encoded_seq, 'ground_truth': inputs[2] if teacher_force else None, 'initial_readout': encoded_seq,
         'states': states})

    model = Model(inputs, [prev_decoded_seq, next_decoded_seq])
    model.encoder = encoder
    model.decoders = [prev_decoder, next_decoder]
    return model


def SkipThoughtModel_new(output_dim, output_length, batch_input_shape=None,
            input_shape=None, batch_size=None, input_dim=None, input_length=None,
            hidden_dim=None, depth=1, broadcast_state=True, unroll=False,
            stateful=False, inner_broadcast_state=True, teacher_force=False,
            peek=False, dropout=0.):

    '''
    Seq2seq model based on [1] and [2]. You can switch between [1] based model and [2]
    based model using the peek argument.(peek = True for [2], peek = False for [1]).
    When peek = True, the decoder gets a 'peek' at the context vector at every timestep.

    Arguments:
    - output_dim : Required output dimension.
    - hidden_dim : The dimension of the internal representations of the model.
    - output_length : Length of the required output sequence.
    - depth : Used to create a deep Seq2seq model. For example, if depth = 3,
                    there will be 3 LSTMs on the enoding side and 3 LSTMs on the
                    decoding side. You can also specify depth as a tuple. For example,
                    if depth = (4, 5), 4 LSTMs will be added to the encoding side and
                    5 LSTMs will be added to the decoding side.
    - broadcast_state : Specifies whether the hidden state from encoder should be
                                      transfered to the deocder.
    - inner_broadcast_state : Specifies whether hidden states should be propogated
                                                    throughout the LSTM stack in deep models.
    - peek : Specifies if the decoder should be able to peek at the context vector
               at every timestep.
    - dropout : Dropout probability in between layers.

    Returns: Keras model to be trained.
    '''

    if isinstance(depth, int):
      depth = (depth, depth)
    if batch_input_shape:
      shape = batch_input_shape
    elif input_shape:
      shape = (batch_size,) + input_shape
    elif input_dim:
      if input_length:
        shape = (batch_size,) + (input_length,) + (input_dim,)
      else:
        shape = (batch_size,) + (None,) + (input_dim,)
    else:
      # TODO Proper error message
      raise TypeError
    if hidden_dim is None:
      hidden_dim = output_dim

    encoder = RecurrentSequential(readout=True, state_sync=inner_broadcast_state,
                                  unroll=unroll, stateful=stateful,
                                  return_states=broadcast_state)
    for _ in range(depth[0]):
      encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], hidden_dim)))
      encoder.add(Dropout(dropout))

    dense1 = TimeDistributed(Dense(hidden_dim))
    dense1.supports_masking = True
    dense2 = Dense(output_dim)

    decoder_next = RecurrentSequential(readout='add' if peek else 'readout_only',
                                  state_sync=inner_broadcast_state, decode=True,
                                  output_length=output_length, unroll=unroll,
                                  stateful=stateful, teacher_force=teacher_force)
    decoder_prev = RecurrentSequential(readout='add' if peek else 'readout_only',
                                  state_sync=inner_broadcast_state, decode=True,
                                  output_length=output_length, unroll=unroll,
                                  stateful=stateful, teacher_force=teacher_force)

    for _ in range(depth[1]):
      decoder_next.add(Dropout(dropout, batch_input_shape=(shape[0], output_dim)))
      decoder_next.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim,
                                  batch_input_shape=(shape[0], output_dim)))

      decoder_prev.add(Dropout(dropout, batch_input_shape=(shape[0], output_dim)))
      decoder_prev.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim,
                                    batch_input_shape=(shape[0], output_dim)))                          

    _input = Input(batch_shape=shape)
    _input._keras_history[0].supports_masking = True
    encoded_seq = dense1(_input)
    encoded_seq = encoder(encoded_seq)
    if broadcast_state:
      assert type(encoded_seq) is list
      states = encoded_seq[-2:]
      encoded_seq = encoded_seq[0]
    else:
      states = None
    encoded_seq = dense2(encoded_seq)
    inputs = [_input]
    if teacher_force:
      truth_tensor_next = Input(batch_shape=(shape[0], output_length, output_dim))
      truth_tensor_next._keras_history[0].supports_masking = True
      truth_tensor_prev = Input(batch_shape=(None, output_length, output_dim))
      truth_tensor_prev._keras_history[0].supports_masking = True
      inputs += [truth_tensor_prev,truth_tensor_next]

    prev_decoded_seq = decoder_prev(encoded_seq,
                          ground_truth=inputs[1] if teacher_force else None,
                          initial_readout=encoded_seq, initial_state=states)

    next_decoded_seq = decoder_next(encoded_seq,
                          ground_truth=inputs[2] if teacher_force else None,
                          initial_readout=encoded_seq, initial_state=states)
    
    model = Model(inputs, [prev_decoded_seq,next_decoded_seq])
    model.encoder = encoder
    model.decoder = [decoder_prev,decoder_next]
    return model