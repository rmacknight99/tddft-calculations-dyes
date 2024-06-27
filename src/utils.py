import autode as ade 
import pandas as pd 
import numpy as np 
from rdkit import Chem
import os, sys
sys.path.append('./src/')
from kws import (
    opt_kws,
    tddft_kws,
    tddft_block,
    exc_opt_block,
    steom_kws,
    steom_block
)

def load_data(file_path):
    df = pd.read_csv(file_path)
    data = [(id, s) for id, s in zip(df['ID'], df['Dye_SMILE'])]
    return data

def update_inp_and_run(calc, block):
    # first print the input file
    calc._executor.generate_input()
    
    # open the input file and add the block at the end
    with open(calc.input.filename, 'a') as f:
        f.write(block)
    
    calc._executor.output.filename = calc._executor.method.output_filename_for(calc._executor)
    calc._executor._execute_external()
    calc._executor.set_properties()
    calc._executor.clean_up()
    
    return calc

def run(id, SMILES, solvent_name='methanol', n_cores=32):
    # get root
    root = os.getcwd()
    # make a directory for the calculation
    os.makedirs(str(id), exist_ok=True)
    # enter
    os.chdir(str(id))
    
    try:
        # make a molecule object
        molecule = ade.Molecule(smiles=SMILES, solvent_name=solvent_name)
        
        # optimize the ground state geometry
        # gs_opt_calc = ade.Calculation(
        #     name='gs_opt', 
        #     molecule=molecule, 
        #     method=ade.methods.ORCA(),
        #     keywords=opt_kws,
        #     n_cores=n_cores
        # )
        # # run the calculation
        # gs_opt_calc.run()
        
        # calculate the absorption spectrum
        tddft_abs_calc = ade.Calculation(
            name='abs', 
            molecule=molecule, 
            method=ade.methods.ORCA(),
            keywords=tddft_kws,
            n_cores=n_cores
        )
        tddft_abs_calc = update_inp_and_run(tddft_abs_calc, tddft_block)
        
        # optimize the excited state geometry
        # exc_opt_calc = ade.Calculation(
        #     name='exc_opt', 
        #     molecule=molecule, 
        #     method=ade.methods.ORCA(),
        #     keywords=opt_kws,
        #     n_cores=n_cores
        # )
        # exc_opt_calc = update_inp_and_run(exc_opt_calc, exc_opt_block)
        
        # calculate the emission spectrum
        # tddft_em_calc = ade.Calculation(
        #     name='em', 
        #     molecule=molecule, 
        #     method=ade.methods.ORCA(),
        #     keywords=tddft_kws,
        #     n_cores=n_cores
        # )
        # tddft_em_calc = update_inp_and_run(tddft_em_calc, tddft_block)
        
    except Exception as e:
        # write file with error
        with open(f'ERROR', 'w') as f:
            f.write(str(e))
        os.chdir(root)
        return 'Failed'

    os.chdir(root)
    return 'Success'