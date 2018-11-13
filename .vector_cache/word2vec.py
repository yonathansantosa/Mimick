# coding: utf-8
from __future__ import division

import struct
import sys
import gzip

import argparse

# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--max', default=3000000)
parser.add_argument('--embdim', default=300)

args = parser.parse_args()

FILE_NAME = "/content/gdrive/My Drive/GoogleNews-vectors-negative300.bin.gz" # outputs GoogleNews-vectors-negative300.bin.gz.txt
SAVE_TO = "./.vector_cache/GoogleNews-vectors-negative300.bin.gz"
MAX_VECTORS = int(args.max) # Top words to take
FLOAT_SIZE = 4 # 32bit float
emb_dim = int(args.embdim)

output_file_name = FILE_NAME + ".txt"

with gzip.open(FILE_NAME, 'rb') as f, open(output_file_name, 'wb') as f_out:
    
    c = None
    
    # read the header
    header = b""
    while c != "\n":
        c = f.read(1)
        header += c

    total_num_vectors, vector_len = (int(x) for x in header.split())
    num_vectors = min(MAX_VECTORS, total_num_vectors)

    print("Taking embeddings of top %d words (out of %d total)" % (num_vectors, total_num_vectors))
    print("Embedding size: %d" % emb_dim)

    for j in xrange(num_vectors):
        word = ""        
        while True:
            c = f.read(1)
            if c == " ":
                break
            word += c

        binary_vector = f.read(FLOAT_SIZE * vector_len)
        txt_vector = [ "%s" % struct.unpack_from('f', binary_vector, i)[0] 
                   for i in xrange(0, len(binary_vector), FLOAT_SIZE) ]

        txt_vector = txt_vector[:emb_dim]
        
        f_out.write("%s %s\n" % (word, " ".join(txt_vector)))
        
        sys.stdout.write("%d%%\r" % ((j + 1) / num_vectors * 100))
        sys.stdout.flush()
        
        if (j + 1) == num_vectors:
            break
            
print("\nDONE!")
print("Output written to %s" % output_file_name)