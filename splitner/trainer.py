import os
import re
from pathlib import Path
from transformers.trainer import Trainer, PREFIX_CHECKPOINT_DIR
# from transformers import Trainer, PREFIX_CHECKPOINT_DIR
from typing import List


class NerTrainer(Trainer):
    def __init__(self, **kwargs):
        super(NerTrainer, self).__init__(**kwargs)

    # don't swap best and last models. Instead maintain ordering and make best model least likely to be removed
    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]
        # glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            best_model_path = checkpoints_sorted[best_model_index]
            del checkpoints_sorted[best_model_index]
            checkpoints_sorted.append(best_model_path)
        return checkpoints_sorted
