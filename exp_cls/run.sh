#!/bin/bash
source activate deepfs
cd /home/kailiangs/open-source-project/deep-feature-select
python -u -m exp_cls.run_all_cls --phase "$1" --device "${2:-cpu}"
