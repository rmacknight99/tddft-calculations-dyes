import autode as ade 
import pandas as pd 
import numpy as np 
from rdkit import Chem
import os, sys, json, time
sys.path.append('./src/')
import cclib
from morfeus.xtb import XTB
from morfeus.conformer import ConformerEnsemble
from kws import ORCA_KWS, GAUSSIAN_KWS

def extract_spectrum_cclib(file_path):
    parser = cclib.io.ccopen(file_path)
    data = parser.parse()
    if hasattr(data, 'etenergies') and hasattr(data, 'etoscs'):
        energies = data.etenergies
        fosc = data.etoscs
    else:
        return pd.DataFrame(columns=['State', 'Energy', 'Wavelength', 'fosc'])
    data_list = []
    for i, (e, f) in enumerate(zip(energies, fosc)):
        data_list.append({
            'State': int(i + 1),
            'Energy': e,
            'Wavelength': 1e7 / e,
            'fosc': f
        })
    df = pd.DataFrame(data_list).round(5)
    return df

def extract_HL_cclib(file_path):
    parser = cclib.io.ccopen(file_path)
    data = parser.parse()
    if hasattr(data, 'moenergies') and hasattr(data, 'homos'):
        homo_idx = data.homos[0]
        moenergies = data.moenergies[0]
        lumo_idx = homo_idx + 1
        gap = (moenergies[homo_idx] - moenergies[lumo_idx]) * -1
    else:
        return {}
    energy_dict = {
        'homo': moenergies[homo_idx],
        'lumo': moenergies[lumo_idx],
        'gap' : gap
    }
    return energy_dict

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

def obabel_geom_gen(SMILES):
    os.system(f'obabel -:"{SMILES}" -oxyz -O init.xyz --gen3d 2>/dev/null')
    return 'init.xyz'

def run_gs_opt(id, SMILES, n_cores, ERROR_MSG, g16=False):
    
    if g16:
        from kws import GAUSSIAN_KWS as KWS
        DFT_METHOD = ade.methods.G16()
    else:
        from kws import ORCA_KWS as KWS
        DFT_METHOD = ade.methods.ORCA()
    
    try:
        ERROR_MSG = "SMILES string error, could not make molecule object"
        molecule_ = ade.Molecule(smiles=SMILES, solvent_name=None)
        charge, mult = molecule_.charge, molecule_.mult
        ERROR_MSG = "openbabel structure generation error"
        if os.path.exists('init.xyz'):
            xyz_file = 'init.xyz'
        else:
            xyz_file = obabel_geom_gen(SMILES)
        ERROR_MSG = "molecule from `init.xyz` error"
        molecule = ade.Molecule(xyz_file, charge=charge, mult=mult, solvent_name=None)
        # see if coordinates are all 0
        if np.all(np.asarray(molecule.coordinates) == 0.):
            molecule_.print_xyz_file(filename='init.xyz')
            molecule = ade.Molecule('init.xyz', charge=charge, mult=mult, solvent_name=None)
        
        # optimize with xTB
        ERROR_MSG = "xTB geometry optimization error"
        os.environ['OMP_NUM_THREADS'] = '1'
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
        ERROR_MSG = "Ground state DFT optimization error"
        gs_opt_calc = ade.Calculation(
            name='gs_opt', 
            molecule=molecule, 
            method=DFT_METHOD,
            keywords=KWS['opt'],
            n_cores=n_cores
        )
        # run the calculation and process
        gs_opt_calc.run()
        
        return True, gs_opt_calc
    except:
        # write file with error
        with open(f'ERROR', 'w') as f:
            f.write(ERROR_MSG)
            
        return False, None

def run_exc_opt(molecule, n_cores, ERROR_MSG, g16=False):
    if g16:
        from kws import GAUSSIAN_KWS as KWS
        DFT_METHOD = ade.methods.G16()
    else:
        from kws import ORCA_KWS as KWS
        DFT_METHOD = ade.methods.ORCA()
        
    try:
        # optimize the excited state geometry
        exc_opt_calc = ade.Calculation(
            name='exc_opt', 
            molecule=molecule, 
            method=DFT_METHOD,
            keywords=KWS['exc_opt'],
            n_cores=n_cores
        )
        # Look for block
        if 'blocks' in KWS:
            exc_opt_block = KWS['blocks']['exc_opt']
        else:
            exc_opt_block = ''
        # run the calculation and process
        exc_opt_calc = update_inp_and_run(exc_opt_calc, exc_opt_block)

        return True, exc_opt_calc
    except:
        # write file with error
        with open(f'ERROR', 'w') as f:
            f.write(ERROR_MSG)
            
        return False, None
    
def run_tddft(molecule, n_cores, ERROR_MSG, use_STEOM=False, g16=False):
    if g16:
        from kws import GAUSSIAN_KWS as KWS
        DFT_METHOD = ade.methods.G16()
    else:
        from kws import ORCA_KWS as KWS
        DFT_METHOD = ade.methods.ORCA()

    try:
        # calculate the absorption spectrum
        tddft_abs_calc = ade.Calculation(
            name='abs', 
            molecule=molecule, 
            method=DFT_METHOD,
            keywords=KWS['tddft'] if not use_STEOM else KWS['tddft_ccsd'],
            n_cores=n_cores
        )
        # run the calculation and process
        if 'blocks' in KWS:
            tddft_block = KWS['blocks']['tddft']
        else:
            tddft_block = ''
        tddft_abs_calc = update_inp_and_run(tddft_abs_calc, tddft_block)
        
        return True, tddft_abs_calc
    except:
        # write file with error
        with open(f'ERROR', 'w') as f:
            f.write(ERROR_MSG)
            
        return False, None

def run(id, SMILES, n_cores=32, use_STEOM=False, run_tddft=True, g16=False):
        
    # get root
    root = os.getcwd()
    # make a directory for the calculation
    os.makedirs(str(id), exist_ok=True)
    # enter
    os.chdir(str(id))

    ERROR_MSG = "Initialization Error"

    try:
        # GROUND STATE OPTIMIZATION
        print(f'Running Ground State Optimization for {SMILES}')
        gs_status, gs_opt_calc = run_gs_opt(id, SMILES, n_cores, ERROR_MSG, g16=g16)
        if gs_status: # If the calculation was scuccessful
            try:
                gs_data = extract_HL_cclib(gs_opt_calc.output.filename)
            except:
                gs_data = extract_HL_gap(gs_opt_calc.output.filename)
        
            # save as JSON
            with open('gs_energies.json', 'w') as f:
                json.dump(gs_data, f, indent=4)
                
            gs_mol = gs_opt_calc.molecule
        else:
            raise Exception('Ground State Optimization Failed')
        
        # EXCITED STATE OPTIMIZATION
        ERROR_MSG = "Excited State Optimization Error"
        print(f'Running Excited State Optimization for {SMILES}')
        exc_status, exc_opt_calc = run_exc_opt(gs_mol, n_cores, ERROR_MSG, g16=g16)
        if exc_status:
            try:
                exc_data = extract_HL_cclib(exc_opt_calc.output.filename)
            except:
                exc_data = extract_HL_gap(exc_opt_calc.output.filename)
            # save as JSON
            with open('exc_energies.json', 'w') as f:
                json.dump(exc_data, f, indent=4)
                
            exc_mol = exc_opt_calc.molecule
        else:
            raise Exception('Excited State Optimization Failed')
        
        if not run_tddft:
            os.chdir(root)
            return 'Success'
        
        # GROUND STATE TD-DFT
        ERROR_MSG = "Ground State TD-DFT Error"
        gs_tddft_status, gs_tddft_calc = run_tddft(gs_mol, n_cores, ERROR_MSG, use_STEOM=use_STEOM, g16=g16)
        if gs_tddft_status:
            if use_STEOM:
                ele_abs_spectrum = extract_spectrum(gs_tddft_calc.output.filename, START='ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS', END='CD SPECTRUM')
            else:
                ele_abs_spectrum = extract_spectrum(gs_tddft_calc.output.filename, START='ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS', END='ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS')
            ele_abs_spectrum.to_csv('ABS.csv', index=False)
        
        # EXCITED STATE TD-DFT
        ERROR_MSG = "Excited State TD-DFT Error"
        exc_tddft_status, exc_tddft_calc = run_tddft(exc_mol, n_cores, ERROR_MSG, use_STEOM=use_STEOM, g16=g16)
        if exc_tddft_status:
            if use_STEOM:
                ele_em_spectrum = extract_spectrum(exc_tddft_calc.output.filename, START='ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS', END='CD SPECTRUM')
            else:
                ele_em_spectrum = extract_spectrum(exc_tddft_calc.output.filename, START='ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS', END='ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS')
            ele_em_spectrum.to_csv('EM.csv', index=False)
        
    except Exception as e:
        # write file with error
        with open(f'ERROR', 'w') as f:
            f.write(str(e))
            f.write('\n')
            f.write(ERROR_MSG)
        os.chdir(root)
        return 'Failed'

    os.chdir(root)
    return 'Success'

def run_low(id, SMILES, solvent_name='methanol', n_cores=64):
     # get root
    root = os.getcwd()
    # make a directory for the calculation
    os.makedirs(str(id), exist_ok=True)
    # enter
    os.chdir(str(id))
    
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
        
        # from the init.xyz, do a conformer search
        if not os.path.exists('crest'):
            _, full_ce = crest_conformer_search(
                os.getcwd() + '/init.xyz',
                solvent=False, 
                take_best=False,
                cores=n_cores,
                quick='--quick'
            )
        else:
            full_ce = ConformerEnsemble.from_crest('crest')
        
        # print(f'Ensemble has {len(full_ce)} conformers')
        
        ELEMENTS = full_ce.elements
        for idx, conformer in enumerate(full_ce):
            # print(f'  Getting HL Gap for Conformer {idx+1}')
            COORDS = conformer.coordinates
            xtb = XTB(ELEMENTS, COORDS)
            # a.u. to eV conversion is 27.21
            AU_TO_EV = 27.21
            HOMO = xtb.get_homo() * AU_TO_EV
            LUMO = xtb.get_lumo() * AU_TO_EV
            gap = (HOMO - LUMO) * -1
            conformer.properties["homo"] = HOMO
            conformer.properties["lumo"] = LUMO
            conformer.properties["gap"] = gap
            # print(f'    HOMO = {HOMO} eV')
            # print(f'    LUMO = {LUMO} eV')
            # print(f'    GAP = {round(gap, 3)} eV')
        
        gs_data = {
            'homo': full_ce.boltzmann_statistic("homo"),
            'lumo': full_ce.boltzmann_statistic("lumo"),
            'gap' : full_ce.boltzmann_statistic("gap")
        }
        with open('gs_energies.json', 'w') as f:
            json.dump(gs_data, f, indent=4)

    except Exception as e:
        # write file with error
        with open(f'ERROR', 'w') as f:
            f.write(str(e))
        os.chdir(root)
        return 'Failed'

    os.chdir(root)
    return 'Success'

def crest_conformer_search(full_path, method=ade.methods.get_lmethod(), solvent=False, cbonds='cbonds', cores=16, take_best=False, opt_confs_with_xtb=False, quick='--quick'):
    """
    Run crest on a molecule to find the lowest energy conformer
    Args:
        full_path (str): full path to the xyz file
        method (autode.wrappers.base.ElectronicStructureMethod): optimization method for crest_best.xyz
        solvent (bool): whether to run crest and optimization in solvent
    """
    if solvent:
        solvent = 'methanol'
    else:
        solvent = None
    # split the path and get the xyz_file and the directory to first enter
    HOME = os.getcwd()
    xyz_file = full_path.split('/')[-1]
    root_dir = '/'.join(full_path.split('/')[:-1])
    if root_dir == '':
        root_dir = '.'
    os.chdir(root_dir)
    # create and enter a directory for running crest
    crest_dir = os.path.join(os.getcwd(), 'crest')
    try:
        os.mkdir(crest_dir)
        os.chdir(crest_dir)
        if solvent:
            crest_cmd = f'crest ../{xyz_file} -T {cores} -niceprint -gfn2//gfnff -noreftopo -{cbonds}  -v3 -alpb {solvent} {quick if quick is not None else ""} > crest_alpb.out'
        else:
            crest_cmd = f'crest ../{xyz_file} -T {cores} -niceprint -gfn2//gfnff -noreftopo -{cbonds} -v3 {quick if quick is not None else ""} > crest.out'
        # run crest command 
        # print(f'  Running command: {crest_cmd}')
        os.system(crest_cmd)
        os.system('cd ..')
    except:
        pass
    
    if take_best:
        os.chdir(crest_dir)
        crest_best_path = os.path.join(os.getcwd(), 'crest_best.xyz')
        m = ade.Molecule(crest_best_path, charge=0, mult=1, solvent_name=solvent)
        m.optimise(method=method)
        m.solvent = "methanol"
        m.name = m.name + '_solvated'
        m.optimise(method=method)
        path_to_optimized = os.path.join(os.getcwd(), m.name + '_optimised_xtb.xyz')
        xtb_opt_paths = [path_to_optimized]
        energies = [m.energy.to("kcal")]
        os.chdir('..')
        m.print_xyz_file()
    else:
        # we have a crest directory with our conformers, time to prune
        ce = ConformerEnsemble.from_crest(crest_dir)
        xtb_opt_paths = []
        energies = []
        xtb_opt_dir = os.path.join(os.getcwd(), 'xtb_opt')
        
        # All conformers more than threshold kcal/mol above the lowest energy conformer are pruned
        ce.prune_energy(threshold=3.0)
        
        # A higher value results in less strict pruning (less different conformers get pruned)
        # A lower value results in more strict pruning (highly similar conformers get pruned)
        ce.prune_rmsd(method="spyrmsd", thres=1.0)
        
        if opt_confs_with_xtb:
            # write the conformer XYZ files
            n_conformers = len(ce.conformers)
            conformer_files = [f"intermediate_conformer_{i+1}.xyz" for i in range(n_conformers)]
            for cf in conformer_files:
                if not os.path.exists(cf):
                    ce.write_xyz(cf, separate=True)
            # take each conformer and optimize it with xTB
            if not solvent:
                solvent = None
            else:
                solvent = "methanol"
            for cf in conformer_files:
                m = ade.Molecule(cf, charge=0, mult=1, solvent_name=solvent)
                os.system(f'mkdir -p {xtb_opt_dir}')
                os.chdir(xtb_opt_dir)
                if cbonds == 'nocbonds':
                    m.optimise(method=method)
                    path_to_optimized = os.path.join(os.getcwd(), m.name + '_optimised_xtb.xyz')
                else:
                    con_m = m.copy()
                    # constrain all H atoms
                    con_m.constraints.cartesian = [idx for idx, atom in enumerate(con_m.atoms) if atom.label == 'H']
                    con_m.optimise(method=method)
                    path_to_optimized = os.path.join(os.getcwd(), m.name + '_optimised_xtb.xyz')
                os.chdir('..')
                xtb_opt_paths.append(path_to_optimized)
                m = ade.Molecule(path_to_optimized, charge=0, mult=1, solvent_name=solvent)
                m.print_xyz_file()
                energies.append(m.energy.to("kcal"))
    os.chdir(HOME)
    return xtb_opt_paths, ce

if __name__ == '__main__':
    # Test cclib parsing
    df = extract_spectrum_cclib('steom/TEST_STEOM_METHANOL/abs_orca.out')
    energy_dict = extract_HL_cclib('steom/TEST_STEOM_METHANOL/gs_opt_orca.out')
    
    print(f'Extracted Absorption Spectrum:\n{df}')
    print(f'Extracted MO Energies and HL gap: {energy_dict}')