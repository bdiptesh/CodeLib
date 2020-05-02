![pylint Score](https://mperlet.github.io/pybadge/badges/10.0.svg)

***

## Table of contents

1. [Introduction](./README.md#introduction)
    1. [Objective](./README.md#objective)
    1. [Project directory structure](./README.md#project-directory-structure)
    1. [Programming style](./README.md#programming-style)
    1. [Version control](./README.md#version-control)
1. [Project documents](./doc)
    1. [Approach](./doc/Approach.pdf) - WIP
1. [Pull request guidelines](./.github/PULL_REQUEST_TEMPLATE.md)
1. [Initial setup](./README.md#initial-setup)
1. [Execution](./README.md#execution)
1. [Unit tests](./README.md#run-unit-tests)
***

## Introduction

#### Objective

The objective of this repository is to:

1. Create a code library/toolkit to automate commonly used machine learning techniques/approaches in a modular environment.
1. Provide best in class approaches developed over a period of time.
1. Reduce development time for machine learning projects.
1. Provide a scalable solution for all machine learning projects.

#### Project directory structure

Please go through our [project directory structure](./doc/Directory_structure.md) here.

#### Programming style

It's good practice to follow accepted standards while coding in python:
1. [PEP 8 standards](https://www.python.org/dev/peps/pep-0008/): For code styles.
1. [PEP 257 standards](https://www.python.org/dev/peps/pep-0257/): For docstrings standards.
1. [PEP 484 standards](https://www.python.org/dev/peps/pep-0484/) For function annotations standards.

Also, it's a good idea to rate all our python scripts with [Pylint](https://www.pylint.org/). If we score anything less than 8/10, we should consider redesigning the code architecture.

A composite pylint ratings for all the codes are automatically computed when we [run the tests](./bin/runTests.sh) and prepended on top of this file.

#### Version control

We use semantic versionning ([SemVer](https://semver.org/)) for version control. You can read about semantic versioning [here](https://semver.org/).

## Initial setup

#### Installation

Extract the project in a local directory. Example:

```console
/home/user/project/CodeLib/
```

Run the following in project directory:

```console
bash install.sh
```

Anaconda python 3.7.3 was used for development of this module.

The code was tested in the following environments.
1. [Anaconda python 3.7.3 64-Bit](https://www.anaconda.com/distribution/)

The python requirements can be found at
1. [Linux 64-Bit](./requirements.txt)
***

## Execution

Run the following in project directory:

```console
python main_module
```

***
## Run unit tests and pylint ratings

To run all unit tests and rate all python scripts, run the following in
project directory:

```console
./bin/runTests.sh
```

Available options:

```console
-a default, runs both code rating and unit tests.
-u unit tests.
-r code rating.
```
The pylint ratings for each python script can be found at
[log/pylint/](./log/pylint/)
***
