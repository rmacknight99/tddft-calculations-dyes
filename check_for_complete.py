import os

complete = 0
total = 0
for root, dirs, files in os.walk('tddft/'):
    if root == 'tddft/':
        continue
    elif os.path.exists(root + '/gs_energies.json'):
        complete += 1
    else:
        # we remove the directory
        print('removing', root)
        os.system('rm -r ' + root)
    total += 1

print('complete:', complete, 'total:', total)
        