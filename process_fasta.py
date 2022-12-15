from Bio import SeqIO


with open("uniprot.fasta", "w") as outputs:    
    for r in SeqIO.parse("uniprot_fasta.fasta", "fasta"):        
        r.id = r.description.split('|')[1]    
        r.description = r.id        
        SeqIO.write(r, outputs, "fasta")