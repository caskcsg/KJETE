**Date**: April 15nd, 2022

**Title**: Code for ICASSP 2022 Paper: [A KNOWLEDGE/DATA ENHANCED METHOD FOR JOINT EVENT AND TEMPORAL RELATION EXTRACTION]

The Raw Dataset Access:The link to download TDDiscourse dataset(Naik et al., 2019) :https://github.com/aakanksha19/TDDiscourse, Link
to download MATRES dataset(Ning et al., 2018): https://github.com/qiangning/MATRES, Link to download TimeBank-Dense (Cassidy et al., 2014) 
dataset: https://github.com/muk343/TimeBankdense.

For convenience, The TB-Dense and MATRES data files are saved in data fold. Download any other necessary files into other/ folder.

1. Before the experiment, modify the configuration file to your own path

2. Data processinng. run python addcommonsense_featurized_data_all_*.py first,the folder  all_joint/ are created

3. Featurize data. run python context_aggregator_*.py. The folder all_context/ are created: . 

the step 2 all_context contains the final files used in the model.

4. Local Model: run python joint_model_addcommonseStandModel_roberta_*.py


For convenience and ease of understanding,You can run it separately on both datasets.