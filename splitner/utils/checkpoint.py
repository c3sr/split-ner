import os
import logging

logger = logging.getLogger(__name__)


def find_best_checkpoint(checkpoint_dir):
    prefix='checkpoint-'
    checkpoints = [chk for chk in os.listdir(checkpoint_dir) if chk.startswith(prefix)]

    print(f'checkpoint_dir = {checkpoint_dir} | checkpoints = {checkpoints}')

    min = float("inf")
    for chk in checkpoints:
        id = int(chk[len(prefix):])
        if id < min:
            min = id
    best_chk = os.path.join(checkpoint_dir, 'checkpoint-' + str(min))
    print(f'Best checkpoint = {best_chk}')
    os.rename(best_chk, os.path.join(checkpoint_dir, f"best_checkpoint-{min}"))

# Sample Usage:
# find_best_checkpoint("../../out/wnut/ner-bert-qa/run-42/checkpoints/")


    


