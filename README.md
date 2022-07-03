# RBF Regression - Houston Housing
This is yet another implementation for the rbf regression problem using the Boston Housing dataset.
It is implemented as part of a homework exercise for [088] Computational Intelligence course 
of ECE Department, AUTh

## Getting Started

### Prerequisites
1. Python (3.6 or higher, preferably 3.9)
2. venv
3. Poetry

To install them on variant Linux distributions follow the instructions below

#### Fedora
```shell
$ sudo dnf upgrade --refresh # updates installed packages and repositories metadata
$ sudo dnf install python python3-pip python3-virtualenv python3-devel poetry
```

#### Ubuntu 
```shell
$ sudo apt update && sudo apt upgrade # updates installed packages and repositories metadata
$ sudo apt install python3 python3-pip python3.9-venv python3.9-dev python3-poetry # ubuntu still offers python2 in its repositories
```

### Running the application
1. 
   a. Create and activate a virtual environment 
   ```shell
   $ python3.9 -m venv venv
   $ source venv/bin/activate
       ```
   b. Use poetry to create and activate your virtual environment
   ```shell
   $ poetry shell
   ```
2. Install necessary python dependencies 
   ```shell
   $ pip install -r requirements.txt # if using pip
   $ poetry install # if using poetry
    ```
