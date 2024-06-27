### Set up

```
git clone https://github.com/rmacknight99/tddft-calculations-dyes.git
cd tddft-calculations-dyes
conda create -n `env_name` python=3.11
pip install -r requirements.txt
```

### Usage

```
python main_tddft.py --help
```

### Update (ORCA) Keywords

see ```kws.py```

### Add Data File

Add a data file with identifiers ('ID' header) and SMILES strings ('SMILES' header)
Specify your file with the ``` --data `filename` ``` option for ```main_tddft.py```