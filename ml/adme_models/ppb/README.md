# DL model to predict PPB (fraction unbound) (WIP). 

In this repo we have a model trained on Chembl data to predict the permeability (Caco-2 A->B Papp) from SMILES.  

In this case we have a lighter DL regression model (GINRegressor) that gets very good performance with the following features implemented to improve training and prediction:

* Simple descriptors such as atom symbols and hybridisation type (one-hot encoded) augmented with Rdkit chemical descriptors which should contribute to lipophilicity (partial charges, aromaticity, vdW radius, etc.). 
* Two linear MLP layers and two convolutional layers.
* A final output layer for the regression (`self.lin`)
* Train / test/ validation splits for training.
* Hyperparameter tuning via grid search with early stopping.
* Evaluation via error statistics (RMSE, MAE and R²).

The code and details of the model featurisation, specification, training and evaluation can be found in `2025-09-21_ml_caco2.ipynb`. The trained model file is `caco2_abpapp_gin.pt` and can be used in any Python workflow.

Training data was retrieved via the Chembl `webresource_client` API, which is pip installable. The retrieved data from Chembl is in `data_cac2_abpapp/caco2_abpapp_chembl.csv` and is comprised of **6081 data points**. See `2025-09-21_retrieve_chembl_caco2.ipynb` for code and further details re: post-processing clean-up of the data.

## Model metrics

```
RMSE: 47.7742
MAE: 18.4958
R²: 0.9062
```

![true vs predicted ic50s](_images/true_vs_predicted_pic50s.png)

Overall good correlation and good (~0.9) performance. Training curves looked okay too. But there are some really big differences in some cases

There's some evidence of model over training in the L2 norms, especially in layers that would be more affected by larger molecules (see below plots). In short, there are some quite large molecules in the dataset, but not many. This may actually cause the model to 'over-react' in training. And it stands to reason that some of the worst predictions are on the largest molecules, where some that are extremely permeable are predicted to be very non-permeable.

Here are some examples (T = True, P = Predicted, Δ = difference between T and P). Note both the molecular size and poor predictions on molecules that are very much permeable.

![Examples of poor predictions](_images/5x_outliers_grid_examples.png)

Looking at the overall fold variation we see the greatest variation in the 2nd convolutional layer, which would be more senstive to the size of the molecule being predicted (see notebook for more detail on this). That said, given we only see it in one fold and overall larger molecules aren't common in the dataset, this is more a reason to be cautious for now. 

![L2 variance](_images/l2_norm_variance.png)
![Per-fold variance](_images/weight_norms_comparison.png)

Given there's large variation in a few variables in one Fold, some measure of normalisation and/or training more on larger molecules is warranted. Certainly the number of really poor predictions is of concern, especially for molecules at the permeable end (Papp < 5). So given the model does well at predicting moderately to very poorly permeable molecules, the performance is probably unacceptable for now and we should aim for R² > 0.95.   

See `2025-09-21_ml_caco2.ipynb` for all the training code and analysis used to generate these data.  

## Next steps

* A detailed post at [my blog](https://ahtheelementofsurprise.wordpress.com/comp-chem-blog/) will step through the model code, construction and training. 
* Experiment with lighter models - maybe we can get similar performance with much faster methods.