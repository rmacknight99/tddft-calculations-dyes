from src.utils import run, load_data
import os, concurrent, tqdm, argparse, psutil, time
import autode as ade 

def get_free_cpus():
    # overspan of 10 seconds take 5 measurements and average
    idle_cpus = 0
    for _ in range(5):
        idle_cpus += (sum(i <= 90.0 for i in psutil.cpu_percent(percpu=True)) // 5)
        time.sleep(2)
    return idle_cpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TDDFT calculations')
    parser.add_argument('--n_cores', type=int, default=64, help='Number of cores to use')
    parser.add_argument('--maxcore', type=int, default=16000, help='Maximum mem per core')
    parser.add_argument('--use_STEOM', action='store_true', help='Use STEOM-DLPNO-CCSD instead of TD-DFT')
    parser.add_argument('--skip_tddft', action='store_true', help='Run the TDDFT calculation')
    parser.add_argument('--data', type=str, default='data/experimental_data.csv', help='Path to the data containing IDs and SMILES')
    parser.add_argument('--solvent_name', type=str, default='methanol', help='Name of the solvent to use')
    parser.add_argument('--debug', type=str, default='False', help='Debug mode, use True/False or an integer index')
    parser.add_argument('--max_processes', type=int, default=None, help='Maximum number of processes to use')
    
    args = parser.parse_args()

    ade.Config.max_core = args.maxcore
    
    # load experimental data
    data = load_data(args.data)
    if args.debug == 'True':
        if args.use_STEOM:
            data = [('TEST_STEOM', 'CC1(/C(N(C)C2=C1C=CC=C2)=C\C=C\C=C\C=C\C3=[N+](C)C(C=CC=C4)=C4C(C)3C)C')]
        else:
            data = [('TEST', 'CO')]
        print(f'Debugging the SMILES string `{data[0][1]}`')
    elif args.debug == 'False':
        pass
    else:
        try:
            debug_idx = int(args.debug) - 1
            data = [data[debug_idx]]
            print(f'Debugging the SMILES string `{data[0][1]}`')
        except:
            print(f'`debug` argument not recognized: {args.debug}...must be an integer index or True/False')
            # now exit
            exit()
        
    # make a directory for the calculations and enter it
    root = os.getcwd()
    if not args.use_STEOM:
        os.makedirs('tddft', exist_ok=True)
        os.chdir('tddft')
    else:
        os.makedirs('steom', exist_ok=True)
        os.chdir('steom')
    
    CORES = args.n_cores
    if args.max_processes is  None:
        free_cpus = get_free_cpus()
    else:
        free_cpus = args.max_processes
    print(f'Number of free CPUs: {free_cpus}')
    CHUNK_SIZE = (free_cpus // CORES)
    print(f'Chunk size: {CHUNK_SIZE}')
    if CHUNK_SIZE < 1:
        print('Warning: Chunk size is less than 1, consider decreasing the number of cores')
        exit()
    
    tasks = data.copy()
    # update tasks to add solvent_name and n_cores args
    tasks = [(id, s, args.solvent_name, CORES, args.use_STEOM, not args.skip_tddft) for id, s in tasks]
    
    pbar = tqdm.tqdm(total=len(tasks), desc='Running Calculations')
    
    # DEBUG mode means no parallel
    if args.debug == 'True':
        for task in tasks:
            result = run(*task)
            pbar.update(1)
        pbar.close()
        print("All calculations are done!")
        # at the end go back to the root directory
        os.chdir(root)
        exit()
    else:
        print("Running calculations in parallel...")
    
        with concurrent.futures.ProcessPoolExecutor(max_workers=CHUNK_SIZE) as executor:
            futures = []
            results = []
            
            # Submit initial tasks
            for task in tasks[:CHUNK_SIZE]:
                future = executor.submit(run, *task)
                futures.append(future)
            
            # As tasks complete, submit new tasks one-by-one
            for i in range(CHUNK_SIZE, len(tasks)):
                # Wait for the first future to complete
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                # Obtain result from the completed future, process it if necessary
                for fut in done:
                    result = fut.result()
                    results.append(result)
                    futures.remove(fut)
                    pbar.update(1)
                # Submit a new task
                future = executor.submit(run, *tasks[i])
                futures.append(future)
                
            # Wait for all remaining tasks to complete
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)
                
        pbar.close()

        print("All calculations are done!")
        
        # at the end go back to the root directory
        os.chdir(root)