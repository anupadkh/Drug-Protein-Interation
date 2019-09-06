#!/bin/sh
#purpose: run PSSM 

psiblast_path="/Users/anupadkh/Downloads/code_data/code_data/ncbi-blast-2.9.0+/bin";
nr_database_path="pssm_db";


fasta="pssm_data"
pssm_output_file_path="pssm_data/output";

for file in `cat $fasta/id_list.txt`;
do 
	#===========================================================================================================
	#Run PSI-BLAST to generate PSSM
	
	if [ -f $pssm_output_file_path/$file.mat ];
	then
		printf "PSSM already exists!!\n";
	else
		printf "running PSI-BLAST...";
		$psiblast_path/psiblast -query $fasta/$file.fasta -db $nr_database_path/protein.fasta -out $pssm_output_file_path/$file.out -num_iterations 3 -num_threads 16 -out_ascii_pssm $pssm_output_file_path/$file.mat;
		printf "...DONE!!!\n";
	fi
	#===========================================================================================================
done
