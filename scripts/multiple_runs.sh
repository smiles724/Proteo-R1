#!/bin/bash

fasta_original_file='examples/fasta.files.native/8hpu_M_N_A.fasta'
fasta_masked_file="examples/fasta.files.design/8hpu_M_N_A/8hpu_M_N_A_CDR_H3.fasta"
pdb_file="examples/pdb.files.native/8hpu_M_N_A.pdb"
output="outputs/maturation/8hpu_M_N_A"

(CUDA_VISIBLE_DEVICES=0 python design.py -f $fasta_masked_file -ag $pdb_file --output $output -ns 100 --fasta_origin $fasta_original_file --run_task affinity_maturation) &
(CUDA_VISIBLE_DEVICES=1 python design.py -f $fasta_masked_file -ag $pdb_file --output $output -ns 100 --fasta_origin $fasta_original_file --run_task affinity_maturation) &
(CUDA_VISIBLE_DEVICES=2 python design.py -f $fasta_masked_file -ag $pdb_file --output $output -ns 100 --fasta_origin $fasta_original_file --run_task affinity_maturation) &
(CUDA_VISIBLE_DEVICES=3 python design.py -f $fasta_masked_file -ag $pdb_file --output $output -ns 100 --fasta_origin $fasta_original_file --run_task affinity_maturation) &
(CUDA_VISIBLE_DEVICES=4 python design.py -f $fasta_masked_file -ag $pdb_file --output $output -ns 100 --fasta_origin $fasta_original_file --run_task affinity_maturation) &
(CUDA_VISIBLE_DEVICES=5 python design.py -f $fasta_masked_file -ag $pdb_file --output $output -ns 100 --fasta_origin $fasta_original_file --run_task affinity_maturation) &
(CUDA_VISIBLE_DEVICES=6 python design.py -f $fasta_masked_file -ag $pdb_file --output $output -ns 100 --fasta_origin $fasta_original_file --run_task affinity_maturation) &
(CUDA_VISIBLE_DEVICES=7 python design.py -f $fasta_masked_file -ag $pdb_file --output $output -ns 100 --fasta_origin $fasta_original_file --run_task affinity_maturation) &
