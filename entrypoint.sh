#!/bin/bash
source ~/.bashrc
pre-commit install
poetry run jupyter-lab --no-browser --port 8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''
