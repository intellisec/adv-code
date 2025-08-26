This directory contains examples which wrap the python scripts into bash scripts for submission to a slurm cluster. At least the partitions need to be adjusted on a per-cluster basis.
The scripts are usually looking for certain environment variables. Check each .sh script to find their names. Common variables are
* BAIT: The name of the bait, e.g. `flask_send_from_directory`
* ATTACKTYPE: E.g. `basic`
* MODEL_VARIANT: E.g. `350M` or
* MODEL: E.g. `Salesforce/codegen-350M-multi`
* TAG: The tag to identify the run, e.g. `flask_basic_some_attempt`. Also identifies the folder with the poisoned samples (see note on ATTACK_TAG)
* ATTACK_TAG: Tag which identifies the poisoned dataset as `$EXPERIMENT_ROOT/attacks/$BAIT/$ATTACKTYPE/$ATTACK_TAG`. Is set to $TAG if not given.
