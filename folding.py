import os

def make_structure(AA_seq, comments='', run_name='test', out_dir='outputs/'):
    first_line = '>' + run_name + comments
    file_name = out_dir+run_name
    with open(file_name+'.fasta', 'w+') as f:
        f.write(first_line+'\n'+AA_seq)

    os.system('omegafold '+file_name+'.fasta '+'outputs/'+file_name)

    
