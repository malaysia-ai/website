from glob import glob
import os

files = glob('*/*.ipynb')

for f in files:
    os.system(f'jupyter nbconvert --to html {f}')