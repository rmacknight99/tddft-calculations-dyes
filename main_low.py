from src.utils import load_data, run_low
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
    parser.add_argument('--data', type=str, default='data/experimental_data.csv', help='Path to the data containing IDs and SMILES')
    parser.add_argument('--solvent_name', type=str, default=None, help='Name of the solvent to use')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()

    ade.Config.max_core = args.maxcore
    
    # load experimental data
    data = load_data(args.data)
    CORES = args.n_cores
    free_cpus = get_free_cpus()
    CHUNK_SIZE = 1
        
    # make a directory for the calculations and enter it
    root = os.getcwd()
    os.makedirs('crest_ensemble', exist_ok=True)
    os.chdir('crest_ensemble')

    tasks = data.copy()
    # update tasks to add solvent_name and n_cores args
    tasks = [(id, s, args.solvent_name, CORES) for id, s in tasks]

    finished_tasks = [i for i in os.listdir('./') if os.path.exists(f'./{i}/gs_energies.json')]
    tasks = [task for task in tasks if f'{task[0]}' not in finished_tasks]
    print(f'Number of tasks to run: {len(tasks)}')
    
    pbar = tqdm.tqdm(total=len(tasks), desc='Running Calculations')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=CHUNK_SIZE) as executor:
        futures = []
        results = []
        
        # Submit initial tasks
        for task in tasks[:CHUNK_SIZE]:
            future = executor.submit(run_low, *task)
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
            future = executor.submit(run_low, *tasks[i])
            futures.append(future)
            
        # Wait for all remaining tasks to complete
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            pbar.update(1)
            
    pbar.close()

    print("All calculations are done!")
    
    # at the end go back to the root directory
    os.chdir(root)