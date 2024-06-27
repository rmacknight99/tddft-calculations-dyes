from src.utils import run, load_data
import os, concurrent, tqdm


if __name__ == "__main__":
    # load experimental data
    data = load_data('data/experimental_data.csv')
    
    # make a directory for the calculations and enter it
    root = os.getcwd()
    os.system(f'rm -rf tddft')
    os.makedirs('tddft', exist_ok=True)
    os.chdir('tddft')
    
    # make a list of tasks
    CORES = 32
    CHUNK_SIZE = (32 // CORES)
    tasks = data.copy()
    
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