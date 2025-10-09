

### PDB Download and Tokenization 

To run examples correctly and deploy your demo locally, please at first download the Foldseek
binary file from [here](https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view?usp=sharing) and place 
it into the `bin` folder. Then add execute permission to the binary file.
```
chmod +x bin/foldseek
```


```markdown
# 1) Download AF2 PDBs for first 10 submitted chunks
python proteindt_pipeline.py download-af2-pdb --base-dir /path/to/proteindt --max-chunks 10

# 2) Build 3Di 
python proteindt_pipeline.py build-3di --base-dir /path/to/proteindt --foldseek-bin ./bin/foldseek --skip-if-exists

# 3) Build worklist (uniprot_id + sequence)
python proteindt_pipeline.py worklist --base-dir /path/to/proteindt --limit 5000

# 4) Assemble final SFT JSONL (sequence + optional 3Di)
python proteindt_pipeline.py assemble --base-dir /path/to/proteindt --chain-policy best

# Or run everything:
python proteindt_pipeline.py all --base-dir /path/to/proteindt --foldseek-bin ./bin/foldseek --max-chunks 10 --skip-if-exists

```



### Protein Inverse Folding

https://github.com/A4Bio/ProteinInvBench/releases/tag/dataset_release


casp15
cath4.2/4.3
mpnn_data
TS500/TS50 


### 

Thermostability

Data Source: https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/meltome/splits.zip
Split: human_cell 

GB1

Data Source: https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/gb1/splits.zip
Split: two_vs_rest 



AAV

Data Source: https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/aav/splits.zip
Split: two_vs_many
Note that the split is not clear in the `set` column



### 

Protein2Text-QA

Data: https://huggingface.co/datasets/tumorailab/Protein2Text-QA
Source: PubMed Central (PMC) articles
Flaws: (1) LLaMA3.1-8B-Instruct (2) Only abstracts 

