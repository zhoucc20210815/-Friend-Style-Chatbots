import torch
import argparse

import torch.nn as nn
import torch.nn.functional as F

from chatbot_src.model import *
from chatbot_src.util import *

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 10

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length, device)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

def run():
    voc = Voc(args.corpus_name)
    # If loading on same machine the model was trained on
    checkpoint = torch.load(args.ckpt)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, args.hidden_size)
    embedding.load_state_dict(embedding_sd)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(args.hidden_size, embedding, args.encoder_n_layers)
    decoder = LuongAttnDecoderRNN(args.attn_model, embedding, args.hidden_size, voc.num_words, args.decoder_n_layers)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)
    print('Models built and ready to go!')

    searcher = GreedySearchDecoder(encoder, decoder)

    evaluateInput(encoder, decoder, searcher, voc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', type=str, default="Friends_ckpt_chatbot/4000_checkpoint.pt")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--attn_model', type=str, default='dot')
    parser.add_argument('--corpus_name', type=str, default='cornell_movie')
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--encoder_n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=3)
    args = parser.parse_args()
    
    device = args.device
    
    run()