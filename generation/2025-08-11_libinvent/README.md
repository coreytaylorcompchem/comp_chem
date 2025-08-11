# Libinvent example

[Libinvent](https://scispace.com/pdf/libinvent-reaction-based-generative-scaffold-decoration-for-4c5iy1phot.pdf) is Reinvent's R-group enumeration tool. In the example here, we use enumerate SMILES strings with wildcards in the `scaffolds.smi` file and dock them with Gnina to limit the enumeration to these locations on the molecule. 

## Deviations from vanilla Reinvent (`staged_learning.toml`)

* Number of epochs; min=20 max=50 (use 500 or more for production)
* Docking with Gnina.
* Scoring function contains physchem for training (QED, MW, etc.).
* Diversity filter (MurckoScaffold).

## Quick notes

* Some files generated were omitted for size reasons; docked results (`.sdf`), some logs (`maize_worfklow_docking.log`) and trained models (`.chkpt`).
* Metrics to track the performance of the model can be found in `02_generation_analysis_and_metrics_ep4.ipynb`. As we've only run for 20 - 50 epochs, performance metrics don't change much (augNLL should trend toward smaller, less negative values as the model improves).
* Some groups will be problematic for Reinvent to parameterise, as Gnina (unfortunately) uses openbabel. Mostly those with ambiguous charge at pH (carboxylic acid). In this case, I have allowed the SMILES starting point to vary in multiple locations, replacing a carboxylic acid.
* Each line in the scaffolds.smi file = a separate GPU process. So if your GPU doesn't have enough RAM, Reinvent will error out.

## TODO

* Add ML model to predict an ADME property (hERG activity).

