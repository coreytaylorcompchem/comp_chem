# DL model to predict Caco-2 activity (WIP). 

In this repo we have a model trained on Chembl data to predict the permeability (Caco-2 A->B Papp) from SMILES.  

In this case we have a lighter DL regression model (GINRegressor) that gets very good performance with the following features implemented to improve training and prediction:

* Simple descriptors such as atom symbols and hybridisation type (one-hot encoded) augmented with Rdkit chemical descriptors which should contribute to lipophilicity (partial charges, aromaticity, vdW radius, etc.). 
* Two linear MLP layers and two convolutional layers.
* A final output layer for the regression (`self.lin`)
* Train / test/ validation splits for training.
* Hyperparameter tuning via grid search with early stopping.
* Evaluation via error statistics (RMSE, MAE and R²).

The code and details of the model featurisation, specification, training and evaluation can be found in `2025-09-21_ml_caco2.ipynb`. The trained model file is `XXX.pt` and can be used in any Python workflow.

Training data was retrieved via the Chembl `webresource_client` API, which is pip installable. The retrieved data from Chembl is in `data_cac2_abpapp/caco2_abpapp_chembl.csv` and is comprised of **6081 data points**. See `2025-09-21_retrieve_chembl_caco2.ipynb` for code and further details re: post-processing clean-up of the data.

## Model metrics

```
RMSE: XXX
MAE: XXX
R²: XXX
```

![true vs predicted ic50s](_images/true_vs_predicted_pic50s.png)



See `2025-09-21_ml_caco2.ipynb` for all the training code and analysis used to generate these data.  

## Next steps

* A detailed post at [my blog](https://ahtheelementofsurprise.wordpress.com/comp-chem-blog/) will step through the model code, construction and training. 
* Experiment with lighter models - maybe we can get similar performance with much faster methods.