# Tasteme
This is the Research Project for the Machine Learning course of the University of Trento. Tasteme aims at suggesting new food products to the end-user, based on what he actually likes. The project has been implemented using Partial Convex Input Neural Networks, the code for the ML model has been found on https://github.com/locuslab/icnn. 
In the report it is possible to find all the steps performed on this project, from the old-code review to the evaluation of the new model. 


## Getting Started
The main code is the icnn_multilabel.py script. Using the proper parameters it is possible to:

* Train and test the ICNN
* Perform a quantitative evaluation of the preference model
* Perform a qualitative evaluation of the preference model.

### Prerequisites

* Python >= 3.5
* Tensorflow 1.8.0
* Scikitlearn
* Numpy
* Pickle
* Matplotlib



## Running the tests

* Train and test the ICNN
python3.5 --save work --nEpoch 100 --testEpoch 10 --model picnn --layers 200 --save_model yes --path_model toKeepSaved/save/ --data ../data/data.also.reverted.pickle --plot_acc yes --db_file ../data/data_no_filters.db 

* Quantitative and Qualitative evaluation (the qualitative has hardcoded product IDs, change them to change the products)

python3.5 --save work --save_model no --path_model toKeepSaved/save/  --data ../data/data.also.reverted.pickle --db_file ../data/data_no_filters.db --test_dir test --test_focus part_of --test_type incremental --test_ratio 10 --test_nutrients 17 24 33 --test_lambda 0.0005


## Authors

* **Marco Mancini** - https://www.linkedin.com/in/marco-mancini-6b2969108/
