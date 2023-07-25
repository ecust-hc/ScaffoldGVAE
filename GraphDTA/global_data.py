global fasta_path



def _init():
    global fasta_path
    fasta_path = '/base'

def set_fasta_path(value):
    global fasta_path
    fasta_path = value

def get_fasta_path():
    global fasta_path
    return fasta_path


