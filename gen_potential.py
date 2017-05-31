#!/usr/bin/env python
'''
Script that generates a custom potential for Wafer.
Should read stdin an print to stdout. There may be stronger coupling
in a future version, but for now this allows the user to test the script
without invoking Wafer each time.

Input should be in the form:

{
    "grid": {
        "x": 50,
        "y": 50,
        "z": 50,
        "dn": 0.01
    }
}

which is a json structure Wafer will send. For testing it's possible to put this
in a file (e.g. test.json) and invoke the script via

cat test.json | ./gen_potential.py

which will print the potiential value at idx if all goes well.

The script can be renamed/copied to anything else. Just don't forget to call wafer with
the -s or --script flag.
'''

from __future__ import print_function

import sys, json;
import numpy as np

def sech(n):
    return 1/np.cosh(n)

# Load data from stdin
data = json.load(sys.stdin)

# This example script calculates the symmetric Poschl-Teller potential, which is
# analytically solvable in one dimension.

# Value of \lambda, can be set by user.
lam = 6

# Find out how deep in {x,y,z} we extend from a central point of {0,0,0}
extent_x = (data["grid"]["dn"]*data["grid"]["x"]-data["grid"]["dn"])/2;
extent_y = (data["grid"]["dn"]*data["grid"]["y"]-data["grid"]["dn"])/2;
extent_z = (data["grid"]["dn"]*data["grid"]["z"]-data["grid"]["dn"])/2;

# Generate a grid to calculate on
sx = np.linspace(-extent_x, extent_x, data["grid"]["x"])
sy = np.linspace(-extent_y, extent_y, data["grid"]["y"])
sz = np.linspace(-extent_z, extent_z, data["grid"]["z"])
x, y, z = np.meshgrid(sx, sy, sz, indexing='ij')

# Compute potiential
coeff = -(lam*(lam+1))/2
V = coeff*(sech(x)*sech(x)) + coeff*(sech(y)*sech(y)) + coeff*(sech(z)*sech(z))

# Output to screen
for outer in V:
    for middle in outer:
        for inner in middle:
            print(inner)

