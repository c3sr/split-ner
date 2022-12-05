import os
import logging

logger = logging.getLogger(__name__)


def find_best_checkpoint(checkpoint_dir):
    prefix='checkpoint-'
    checkpoints = [chk for chk in os.listdir(checkpoint_dir) if chk.startswith(prefix)]

    print(f'checkpoint_dir={checkpoint_dir}   checkpoints={checkpoints}')

    min = 2**32
    for chk in checkpoints:
        id = int(chk[len(prefix):])
        if id < min:
            min=id
    best_chk = os.path.join(checkpoint_dir, 'checkpoint-'+str(min))
    print('Best checkpoint =', best_chk)
    os.rename(best_chk, os.path.join(checkpoint_dir, "best_checkpoint"))

#find_best_checkpoint("/mnt/sdc/workspace/sec-ner/models/security/ner-bert-qa/checkpoints/")


    


