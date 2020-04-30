***

## Table of contents

1. [Guidelines](./README.md#guidelines)
    1. [Objective](./README.md#objective)
    1. [Project structure](./README.md#project-structure)
    1. [Programming style](./README.md#programming-style)
    1. [Version control](./README.md#version-control)
***

## Guidelines

| Fast | Slow |
|----|----|
| Local variables | Global variables |
| map | for loops |

#### Project structure

This repository provides a sample structure of a project. Since we would like to have a common structure for all our projects, the structure should be able to scale with large of applications with internal packages.

In larger applications, we may have one or more internal packages that are either tied together with a wrapper shell script or that provide specific functionality to a larger library we are packaging. We will lay out the conventions to accommodate for this:

```
project_name/
│
├── bin/
│   ├── hiveQueries.sh
│   └── runTests.sh
│
├── data/
│   ├── input/
│   │   ├── raw_data.csv
│   │   └── input.csv
│   └── output/
│       ├── model_output.csv
│       └── model_diagnostics.csv
│
├── doc/
│   ├── problem_statement.md
│   ├── approach.pdf
│   └── latex/
│
├── hive/
│   ├── hive_query_1.hql
│   └── hive_query_2.hql
│
├── log/
│   ├── hive_queries.out
│   ├── main_module.out
│   └── pylint/
│       ├── main_module-__init__-py.out
│       ├── main_module-__main__-py.out
│       └── pylint.out
│
├── main_module/
│   ├── __init__.py
│   ├── __main__.py
│   └── lib/
│       ├── metrics.so
│       ├── cfg.py
│       ├── stat.py
│       ├── opt.py
│       ├── utils.py
│   	└── tmp/
│           ├── build/
│           ├── metrics.pyx
│           ├── metrics.so
│           ├── metrics.c
│           ├── setup.py
│           └── build.sh
│
├── test/
│   ├── __init__.py
│   ├── test_stat.py
│   └── test_opt.py
│
├── install.sh
├── LICENSE
├── README.md
└── requirements.txt
```
