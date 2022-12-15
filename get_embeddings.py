# !pip install torch transformers sentencepiece h5py

# #@title Set up working directories and download files/checkpoints. { display-mode: "form" }
# # Create directory for storing model weights (2.3GB) and example sequences.
# # Here we use the encoder-part of ProtT5-XL-U50 in half-precision (fp16) as 
# # it performed best in our benchmarks (also outperforming ProtBERT-BFD).
# # Also download secondary structure prediction checkpoint to show annotation extraction from embeddings
# !mkdir protT5 # root directory for storing checkpoints, results etc
# !mkdir protT5/protT5_checkpoint # directory holding the ProtT5 checkpoint
# !mkdir protT5/subcell_checkpoint # directory storing the supervised classifier's checkpoint
# !mkdir protT5/output # directory for storing your embeddings & predictions
# !wget -nc -P protT5/ https://rostlab.org/~deepppi/example_seqs.fasta
# !wget -nc -P protT5/protT5_checkpoint https://rostlab.org/~deepppi/protT5_xl_u50_encOnly_fp16_checkpoint/pytorch_model.bin
# !wget -nc -P protT5/protT5_checkpoint https://rostlab.org/~deepppi/protT5_xl_u50_encOnly_fp16_checkpoint/config.json
# # Huge kudos to the bio_embeddings team here! We will integrate the new encoder, half-prec ProtT5 checkpoint soon
# !wget -nc -P protT5/subcell_checkpoint http://data.bioembeddings.com/public/embeddings/feature_models/t5/subcell_checkpoint.pt

# In the following you can define your desired output. Current options:
# per_residue embeddings
# per_protein embeddings
# secondary structure predictions

# Replace this file with your own (multi-)FASTA
# Headers are expected to start with ">";
seq_path = "uniprot_fasta.fasta"

# whether to retrieve embeddings for each residue in a protein 
# --> Lx1024 matrix per protein with L being the protein's length
# as a rule of thumb: 1k proteins require around 1GB RAM/disk
per_residue = True 
per_residue_path = "./protT5/output/uniprot_embeddings.h5" # where to store the embeddings

# whether to retrieve per-protein embeddings 
# --> only one 1024-d vector per protein, irrespective of its length
per_protein = False
per_protein_path = "./protT5/output/per_protein_embeddings.h5" # where to store the embeddings

# whether to retrieve secondary structure predictions
# This can be replaced by your method after being trained on ProtT5 embeddings
subcell_mem = True
subcell_path = "./protT5/output/subcell.csv" # file for storing predictions
mem_path = "./protT5/output/membrane.csv" # file for storing predictions
# make sure that either per-residue or per-protein embeddings are stored
assert per_protein is True or per_residue is True or subcell_mem is True, print(
    "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")

#@title Import dependencies and check whether GPU is available. { display-mode: "form" }
from transformers import T5EncoderModel, T5Tokenizer
import torch
from torch import nn
import h5py
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))

#@title Network architecture for subcell. loc. prediction and Membrane-bound pred. { display-mode: "form" }
# Feed forward neural network to predict a) subcellular localization and 
# b) classifies membrane-bound from water-soluble proteins
class FNN( nn.Module ):
    
    def __init__( self ):
        super(FNN, self).__init__()
        # Linear layer, taking embedding dimension 1024 to make predictions:
        self.layer = nn.Sequential(
                        nn.Linear( 1024, 32),
                        nn.Dropout( 0.25 ),
                        nn.ReLU(),
                        )
        # subcell. classification head
        self.loc_classifier = nn.Linear( 32, 10)
        # membrane classification head
        self.mem_classifier = nn.Linear( 32,  2)

    def forward( self, x):
        # Inference
        out = self.layer( x ) 
        Yhat_loc = self.loc_classifier(out)
        Yhat_mem = self.mem_classifier(out)
        return Yhat_loc, Yhat_mem 

#@title Load the checkpoint for secondary structure prediction. { display-mode: "form" }
def load_subcell_model():
  checkpoint_dir="./protT5/subcell_checkpoint/subcell_checkpoint.pt"
  state = torch.load( checkpoint_dir )
  model = FNN()
  model.load_state_dict(state['state_dict'])
  model = model.eval()
  model = model.half()
  model = model.to(device)
  return model


#@title Load ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50) 
def get_T5_model():
    model = T5EncoderModel.from_pretrained("./protT5/protT5_checkpoint/", torch_dtype=torch.float16)
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False ) 

    return model, tokenizer


#@title Read in file in fasta format. { display-mode: "form" }
def read_fasta( fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    seqs = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                #my line
                uniprot_id = line.split('|')[1]
                seqs[ uniprot_id ] = ''
                print('uniprot id', uniprot_id)
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq= ''.join( line.split() ).upper().replace("-","")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq 
    example_id=next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id,seqs[example_id]))

    return seqs

#@title Generate embeddings. { display-mode: "form" }
# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings( model, tokenizer, seqs, per_residue, per_protein, subcell_mem, 
                   max_residues=4000, max_seq_len=1000, max_batch=100 ):

    if subcell_mem:
      subcell_model = load_subcell_model()

    results = {"residue_embs" : dict(), 
               "protein_embs" : dict(),
               "subcell" : dict(),
               "mem" : dict(), 
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim  
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]

                if subcell_mem: # in case you want to predict secondary structure from embeddings
                  subcell_Yhat, mem_Yhat = subcell_model(emb.mean(dim=0,keepdims=True))
                  results["subcell"][identifier] = torch.max( subcell_Yhat, dim=1)[1].detach().cpu().numpy().squeeze()
                  results["mem"][identifier] = torch.max( mem_Yhat, dim=1)[1].detach().cpu().numpy().squeeze()


                if per_residue: # store per-residue embeddings (Lx1024)
                    results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()


    passed_time=time.time()-start
    avg_time = passed_time/len(seq_dict)
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results

#@title Write embeddings to disk. { display-mode: "form" }
def save_embeddings(emb_dict,out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    return None

#@title Write predictions to disk. { display-mode: "form" }
def write_prediction_csv(predictions, out_path, mode):
  # Label mapping for subcellular localization
  subcell_mapping = {
      0: "Cell_Membrane",
      1: "Cytoplasm",
      2: "Endoplasmatic Reticulum",
      3: "Golgi Apparatus",
      4: "Lysosome or vacuole",
      5: "Mitochondrion",
      6: "Nucleus",
      7: "PEROXISOME",
      8: "Plastid",
      9: "Extracellular"
  }
  # Label mapping for membrane-bound
  mem_mapping = {
      0: "Soluble",
      1: "Membrane-bound"
  }

  if mode=="subcell":
    class_mapping=subcell_mapping
  elif mode=="mem":
    class_mapping=mem_mapping
  else:
    raise NotImplemented

  with open(out_path, 'w+') as out_f:
      out_f.write( '\n'.join( 
          [ "{},{}".format( 
              seq_id, class_mapping[int(yhat)]) 
          for seq_id, yhat in predictions.items()
          ] 
            ) )
  return None

# Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
model, tokenizer = get_T5_model()



# Load example fasta.
seqs = read_fasta( seq_path )

# Compute embeddings and/or secondary structure predictions
results = get_embeddings( model, tokenizer, seqs,
                         per_residue, per_protein, subcell_mem)

print(results)

# Store per-residue embeddings
if per_residue:
  save_embeddings(results["residue_embs"], per_residue_path)
if per_protein:
  save_embeddings(results["protein_embs"], per_protein_path)
if subcell_mem:
  print("Start writing predictions")
  write_prediction_csv(results["subcell"], subcell_path,mode="subcell")
  write_prediction_csv(results["mem"], mem_path, mode="mem")