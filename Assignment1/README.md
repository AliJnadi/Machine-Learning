# M21-RO-ML

This project is for Assignment1 solution related to subject of Machine Learning for the 1st year Master 2021 Innopolis University, Russia.


## Prerequisites

* matplotlib>=3.4.3
* numpy>=1.21.2
* pandas>=1.3.3
* scikit_learn>=1.0
* scipy>=1.7.1

**All the libraries can be pip installed** using `pip install -r requirements.txt`

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
1. Navigate to repository folder
1. Install dependencies which are specified in requirements.txt. use `pip install -r requirements.txt` or `pip3 install -r requirements.txt`
1. Run `database_prepare.py` if you want to split database to train and test sets.
1. Run `outliers_filtering.py` if you want to filter train dataset from outliers the result will be saved to new data set called `test_o.cvs`.
1. Run `model.py` if you want to train and test the model.

## Repository Structure

```tree
├── .gitignore                               <- Files that should be ignored by git.
│                               
├── requirements.txt                         <- The requirements file for reproducing the analysis environment, e.g.
│                                               generated with `pip freeze > requirements.txt`. Might not be needed if using conda.
│
├── Data                                     <- Data files directory
│   ├── database                             <- main database directory
│   │        └── flight_delay.csv            <- main csv file
│   │
|   ├── test_set                             <- test data directory
│   │        └── test.csv                    <- test csv file
|   |
|   └── train_set                            <- train data directory
|            ├── train.csv                   <- train csv file without filtering
│            └── train_o.csv                 <- train csv file with filtering
|
└── src                                      <- Code for use in this project.
    ├── database_prepare.py                  <- database preparing script
    ├── model.py                             <- model creation, training and testing script
    └── outliers_filtering.py                <- training data filtering using Isolating forests script
```
## Contributing to This Repository
Contributions to this repository are greatly appreciated and encouraged.<br>
To contribute an update simply:
* Submit an issue describing your proposed change to the repo in question.
* The repo owner will respond to your issue promptly.
* Fork the desired repo, develop and test your code changes.
* Edit this document and the template README.md if needed to describe new files or other important information.
* Submit a pull request.

## Contact
Email: a.jnadi@innopolis.university
