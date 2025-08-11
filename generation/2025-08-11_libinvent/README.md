# Libinvent example

## A couple of quick notes for running Libinvent

* Some groups will be problematic for Reinvent to parameterise, as Gnina (unfortunately) uses openbabel. Mostly those with ambiguous charge at pH (carboxylic acid). In this case, I have allowed the SMILES starting point to vary in multiple locations, replacing a carboxylic acid.
* Each line in the scaffolds.smi file = a separate GPU process. So if your GPU doesn't have enough RAM, Reinvent will error out.
