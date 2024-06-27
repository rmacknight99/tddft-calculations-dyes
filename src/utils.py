import autode as ade 
import pandas as pd 
import numpy as np 
from rdkit import Chem
import os, sys, json
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

def extract_spectrum(fname, START='ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS', END='ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS'):
    # open the output file
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
        
    # find line index for TITLE
    start_idx = lines.index(START) + 5
    end_idx = lines.index(END) - 2
    abs_lines = lines[start_idx:end_idx]
    headers = 'State   Energy    Wavelength   fosc        P2         PX        PY        PZ'.strip().split()
    units = ['', 'cm**-1', 'nm', '', 'au**2', 'au', 'au', 'au']
    headers_with_units = [f'{h} {u}'.strip() for h, u in zip(headers, units)]
    # make a dataframe
    data = []
    for l in abs_lines:
        vals = l.split()
        data_dict = {h: float(v) for h, v in zip(headers, vals)}
        data.append(data_dict)

    return pd.DataFrame(data)

def extract_HL_gap(fname):
    energies = {'homo': None, 'lumo': None, 'gap': None}
    search = 'ORBITAL ENERGIES'
    # open the output file
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    
    start_idx = lines.index(search) + 4
    end_idx = lines.index('MULLIKEN ATOMIC CHARGES') - 6
    
    occ_lines = lines[start_idx:end_idx]
    headers = lines[start_idx-1].strip().split()
    data = []
    for l in occ_lines:
        vals = [float(v) for v in l.strip().split()]
        data_dict = {h: v for h, v in zip(headers, vals)}
        data.append(data_dict)
    df = pd.DataFrame(data)
    # Find where first index of df['OCC'] == 0
    lumo_idx = df[df['OCC'] == 0].index[0]
    homo_idx = lumo_idx - 1
    lumo_energy, homo_energy = df['E(eV)'].tolist()[lumo_idx], df['E(eV)'].tolist()[homo_idx]
    gap = (homo_energy - lumo_energy) * -1
    energies['homo'], energies['lumo'], energies['gap'] = lumo_energy, homo_energy, gap
    return energies

def obabel_geom_gen(SMILES):
    os.system(f'obabel -:"{SMILES}" -oxyz -O init.xyz --gen3d 2>/dev/null')
    return 'init.xyz'
    

def run(id, SMILES, solvent_name='methanol', n_cores=32, use_STEOM=False):
    # get root
    root = os.getcwd()
    # make a directory for the calculation
    os.makedirs(str(id), exist_ok=True)
    # enter
    os.chdir(str(id))
    if use_STEOM:
        tddft_kws = steom_kws
        tddft_block = steom_block
    
    try:
        # make a molecule object
        molecule_ = ade.Molecule(smiles=SMILES, solvent_name=solvent_name)
        charge, mult = molecule_.charge, molecule_.mult
        # generate initial openbabel geometry
        if os.path.exists('init.xyz'):
            xyz_file = 'init.xyz'
        else:
            xyz_file = obabel_geom_gen(SMILES)
        molecule = ade.Molecule(xyz_file, charge=charge, mult=mult, solvent_name=solvent_name)
        # see if coordinates are all 0
        if np.all(np.asarray(molecule.coordinates) == 0.):
            molecule_.print_xyz_file(filename='init.xyz')
            molecule = ade.Molecule('init.xyz', charge=charge, mult=mult, solvent_name=solvent_name)
        
        # optimize with xTB
        before_opt_coords = molecule.coordinates
        xtb_opt = ade.Calculation(
            name='gs_opt_low',
            molecule=molecule,
            method=ade.methods.XTB(),
            keywords=ade.methods.XTB().keywords.opt,
            n_cores=n_cores
        )
        xtb_opt.run()
        molecule.print_xyz_file(filename='xtb_opt.xyz')
        
        # optimize the ground state geometry
        gs_opt_calc = ade.Calculation(
            name='gs_opt', 
            molecule=molecule, 
            method=ade.methods.ORCA(),
            keywords=opt_kws,
            n_cores=n_cores
        )
        # run the calculation and process
        gs_opt_calc.run()
        gs_data = extract_HL_gap(gs_opt_calc.output.filename)
        # save as JSON
        with open('gs_energies.json', 'w') as f:
            json.dump(gs_data, f, indent=4)
        
        # calculate the absorption spectrum
        tddft_abs_calc = ade.Calculation(
            name='abs', 
            molecule=molecule, 
            method=ade.methods.ORCA(),
            keywords=tddft_kws,
            n_cores=n_cores if not use_STEOM else 16
        )
        # run the calculation and process
        tddft_abs_calc = update_inp_and_run(tddft_abs_calc, tddft_block)
        if use_STEOM:
            ele_abs_spectrum = extract_spectrum(tddft_abs_calc.output.filename, START='ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS', END='CD SPECTRUM')
            ele_abs_spectrum.to_csv('ELE_ABS.csv', index=False)
        else:
            ele_abs_spectrum = extract_spectrum(tddft_abs_calc.output.filename, START='ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS', END='ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS')
            vel_abs_spectrum = extract_spectrum(tddft_abs_calc.output.filename, START='ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS', END='CD SPECTRUM')
            ele_abs_spectrum.to_csv('ELE_ABS.csv', index=False)
            vel_abs_spectrum.to_csv('VEL_ABS.csv', index=False)
        
        # optimize the excited state geometry
        exc_opt_calc = ade.Calculation(
            name='exc_opt', 
            molecule=molecule, 
            method=ade.methods.ORCA(),
            keywords=opt_kws,
            n_cores=n_cores
        )
        # run the calculation and process
        exc_opt_calc = update_inp_and_run(exc_opt_calc, exc_opt_block)
        exc_data = extract_HL_gap(exc_opt_calc.output.filename)
        # save as JSON
        with open('exc_energies.json', 'w') as f:
            json.dump(exc_data, f, indent=4)
        
        # calculate the emission spectrum
        tddft_em_calc = ade.Calculation(
            name='em', 
            molecule=molecule, 
            method=ade.methods.ORCA(),
            keywords=tddft_kws,
            n_cores=n_cores if not use_STEOM else 16
        )
        tddft_em_calc = update_inp_and_run(tddft_em_calc, tddft_block)
        if use_STEOM:
            ele_em_spectrum = extract_spectrum(tddft_em_calc.output.filename, START='ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS', END='CD SPECTRUM')
            ele_em_spectrum.to_csv('ELE_EM.csv', index=False)
        else:
            ele_em_spectrum = extract_spectrum(tddft_em_calc.output.filename, START='ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS', END='ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS')
            vel_em_spectrum = extract_spectrum(tddft_em_calc.output.filename, START='ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS', END='CD SPECTRUM')
            ele_em_spectrum.to_csv('ELE_EM.csv', index=False)
            vel_em_spectrum.to_csv('VEL_EM.csv', index=False)
        
    except Exception as e:
        # write file with error
        with open(f'ERROR', 'w') as f:
            f.write(str(e))
        os.chdir(root)
        return 'Failed'

    os.chdir(root)
    return 'Success'