#!/bin/bash

rm -rf dist/
~/.conda/envs/gwjax311/bin/python3.11 -m build
~/.conda/envs/gwjax311/bin/python3.11 -m twine upload dist/*

