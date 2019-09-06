#!/bin/sh
#purpose: make PSSM db

psiblast_path="/Users/anupadkh/Downloads/code_data/code_data/ncbi-blast-2.9.0+/bin";
nr_database_path="pssm_db";


fasta="pssm_data"
pssm_output_file_path="pssm_data/output";

$psiblast_path/makeblastdb -in $nr_database_path/protein.fasta -dbtype prot