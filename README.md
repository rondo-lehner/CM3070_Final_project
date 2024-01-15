CM3070_Final_project
==============================

UoL BSc Computer Science Final Project - Land cover classification on Deep Globe 2018 dataset

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Reminder when using new EC2 instance
* Move AWS keypair: {file_name}.pem file to ~/.ssh/ (chmod 400 {file_name}.pem)
* Install docker engine: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository 
    * Don't forget post installation steps: https://docs.docksudo usermod -aG docker $USERer.com/engine/install/linux-postinstall/
* Install unzip `sudo apt install unzip` for aws cli installation
* Install latest version of aws-cli: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
* Configure aws cli: https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html
* Login to ECR: https://github.com/aws/deep-learning-containers 
    * `aws ecr get-login-password --region eu-central-2 --profile [PROFILE_NAME] | docker login --username AWS --password-stdin 380420809688.dkr.ecr.eu-central-2.amazonaws.com`
* `docker pull 380420809688.dkr.ecr.eu-central-2.amazonaws.com/tensorflow-training:2.13.0-cpu-py310-ubuntu20.04-ec2`
* Install screen inside the running container `apt install screen`

