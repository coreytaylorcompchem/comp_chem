# Reinvent example with multiple ML prediction models added to the scoring function 

[Reinvent](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00812-5) is a molecule generation tool that uses Reinforcement Learning (RL) to generate compounds. In the example here, we use a scaffold from a from Chembl ligand (scaffolds.smi) to guide R-group enumeration. Metrics to track the performance of the model can be found in `02_generation_analysis_and_metrics_ep4.ipynb`.

In this example, we have added two trained ADME models to the scoring function;

* hERG classification (1=active, 0=inactive)
* LogD regression model to directly predict LogD
* CYP3A4 regression model to predict pIC50 against the isoform as a proxy for others important isoforms.

Generated molecules are scored by this model and the results are used to re-train the Reinvent RL.

The run can be started with the following command after activating your Reinvent conda environment:

`reinvent -l log.log staged_learning.toml`

Some relevant information about the data transforms of scoring function items in the `staged_learning.toml` can be found in the [tuning Reinvent notebook](https://github.com/coreytaylorcompchem/comp_chem/blob/main/generation/2025-08-29_tuning_reinvent.ipynb)

## Deviations from vanilla Reinvent (`staged_learning.toml`)

* Number of epochs; min=20 max=20 (strongly recommend using 500 or more for production)
* Docking with Gnina.
* Scoring function contains physchem for training (QED, MW, etc.).
* Diversity filter (MurckoScaffold).
* Addition of a ML predictive models (hERG and LogD) to the scoring function.
    * Notebooks with development and training code for the hERG and LogD models, as well as the model files themselves, can be found in this GH at 
        * [/comp_chem/ml/adme_models/herg](https://github.com/coreytaylorcompchem/comp_chem/tree/main/ml/adme_models/herg).
        * [/comp_chem/ml/adme_models/logd](https://github.com/coreytaylorcompchem/comp_chem/tree/main/ml/adme_models/logd).
        * [/comp_chem/ml/adme_models/cyp_3a4](https://github.com/coreytaylorcompchem/comp_chem/tree/main/ml/adme_models/cyp_3a4).

## Quick notes

* Some files generated were omitted for size reasons; docked results (`.sdf`), some logs (`maize_worfklow_docking.log`) and trained models (`.chkpt`).
* Metrics to track the performance of the model can be found in `02_generation_analysis_and_metrics_ep4.ipynb`. As we've only run for 20 - 50 epochs, performance metrics don't change much (augNLL should trend toward smaller, less negative values as the model improves).
* All the parts of the scoring function for Reinvent like physchem QED, etc. are termed **components** in the Reinvent world. Addition of the hERG ML model (or any model) to the scoring function necessitates writing your own component file to register and interact with the Reinvent workflow and make the prediction which is added to the output `.csv` Working examples can be found in `components/comp_herg.py` and `component/comp_logd.py` and `component/comp_cyp3a4.py`. 
    * You can set this up wherever you like but it's just easier to dump them in with all the other components where Reinvent is installed (usually something like `~/miniconda3/envs/reinvent4/lib/python3.10/site-packages/reinvent_plugins/components/`)
    * Just bear in mind if you build/use different models to mine, you'll need to re-write the component file. The examples in `components` should be a decent guide.
    * To work with Reinvent naming conventions you **must** name your component file with the convention comp_[YOUR MODEL NAME].py 
    * A good quick check to make sure any component file you write has worked is run `python reinvent_component_test.py`. This will display registered components and classes and confirm that your model is available to Reinvent. 

## TODO

* Blog post explaining all this mess.

