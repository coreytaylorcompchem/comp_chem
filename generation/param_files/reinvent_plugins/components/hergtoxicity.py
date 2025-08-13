# *********************************************
# File: /.../reinvent_plugins/components/hergtoxicity.py
# *********************************************

# 1. Component declaration required by REINVENT
component_type = "hergtoxicity"
parameters_type = dict

# 2. Delayed import of BaseScoringComponent to avoid early REINVENT discovery issues
def _get_base():
    from reinvent.scoring.score_components import BaseScoringComponent
    return BaseScoringComponent

class HERGToxicityComponent(_get_base()):
    def __init__(self, parameters: dict):
        super().__init__(parameters)

    def calculate_score(self, molecules):
        # Handle plain SMILES strings
        if len(molecules) and isinstance(molecules[0], str):
            smiles_list = molecules
        else:
            smiles_list = [mol.smiles for mol in molecules]

        # Dummy scoring: just return 0.5 for each
        return [0.5 for _ in smiles_list]

if __name__ == "__main__":
    print("Module loaded OK")

def get_component():
    return HERGToxicityComponent, None  # Or replace None with your parameter class if you have one

