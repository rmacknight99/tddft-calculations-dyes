from src.utils import run, load_data
import os, concurrent, tqdm, argparse, psutil, time
import autode as ade 

def get_free_cpus():
    # overspan of 10 seconds take 5 measurements and average
    idle_cpus = 0
    for _ in range(5):
        idle_cpus += (sum(i <= 50.0 for i in psutil.cpu_percent(percpu=True)) // 5)
        time.sleep(2)
    return idle_cpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TDDFT calculations')
    parser.add_argument('--n_cores', type=int, default=64, help='Number of cores to use')
    parser.add_argument('--maxcore', type=int, default=16000, help='Maximum mem per core')
    parser.add_argument('--use_STEOM', action='store_true', help='Use STEOM-DLPNO-CCSD instead of TD-DFT')
    parser.add_argument('--data', type=str, default='data/experimental_data.csv', help='Path to the data containing IDs and SMILES')
    parser.add_argument('--solvent_name', type=str, default='methanol', help='Name of the solvent to use')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()

    ade.Config.max_core = args.maxcore
    
    # load experimental data
    data = load_data(args.data)
    if args.debug:
        if args.use_STEOM:
            data = [('TEST_STEOM', 'CO')]
        else:
            data = [('TEST', 'CO')]
    
    # make a directory for the calculations and enter it
    root = os.getcwd()
    if not args.use_STEOM:
        os.makedirs('tddft', exist_ok=True)
        os.chdir('tddft')
    else:
        os.makedirs('steom', exist_ok=True)
        os.chdir('steom')
    
    CORES = args.n_cores
    free_cpus = get_free_cpus()
    print(f'Number of free CPUs: {free_cpus}')
    CHUNK_SIZE = (free_cpus // CORES)
    print(f'Chunk size: {CHUNK_SIZE}')
    
    tasks = data.copy()
    # update tasks to add solvent_name and n_cores args
    tasks = [(id, s, args.solvent_name, CORES, args.use_STEOM) for id, s in tasks]
    
    pbar = tqdm.tqdm(total=len(tasks), desc='Running Calculations')
    
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