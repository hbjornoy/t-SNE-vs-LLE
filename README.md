#### Project in Advanced Machine Learning, MICRO-570, EPFL, Lausanne
#### Håvard Bjørnøy and Hedda H. B. Vik

# Comparison of the Locally Linear Embedding (LLE) and t-distributed Stochastic Neighbourhood Embedding (t-SNE). 
This repository contains a project in Advanced Machine Learning, MICRO-570 at EPFL. 

To run ``main.py``, please do the following: 
1. If needed, install requirements (se 'Installments')
2. Make sure that you have all files belonging to the original Zip file in their original place. 
3. Run ``main.py`` from its original place
4. You must exit a plot-window before pressing enter in the terminal for it to work properly.


### Installments: 
The following programs and libraries are used in this project: 
- Python 3.6.1
	* probably works with later versions, if not install via https://www.python.org/downloads/release/python-361/
- numpy
	* ``pip install numpy
- sklearn
	* ``pip install scikit-learn``
- seaborn
	* ``pip install seaborn``
- matplotlib
	* ``pip install matplotlib``
- ipywidgets
	* ``pip install ipywidgets``
- pickle
	* A part of standard Python 3.6.1
- time 
	* A part of standard Python 3.6.1
- mpl_toolkits
	* if it doesnt recognise module, then upgrade matplotlib with: ``pip install --upgrade matplotlib``

if you do not have pip, get pip by following these instructions:
	* https://pip.pypa.io/en/stable/installing/

In addition, we have the following import: 
- ``from mpl_toolkits.mplot3d import Axes3D``


### Content: 
This repository contains the following items: 
#### PDFs:
- Report.pdf: The report. 
#### Python files: 
- ``main.py``
	* A summary of everything that is done in this project. 
- ``helpers.py``
	* Contains simple help functions
- ``pickle_functions.py``
	* Contains functions to create or load pickles of transformations
- ``plot_functions.py``
	* Contains functions used to make plots
- ``plot_mnist.py``
	* Contains a function used to plot MNIST
	
#### Jupyter notebooks: 
- ``Section_III_B-1.ipynb``
- ``Section_III_B-2.ipynb``
- ``Section_III_C.ipynb``
- ``Section_III_D.ipynb``
- ``Section_III_E.ipynb``
- ``Section_III_F.ipynb``
- ``Section_IV.ipynb``

We invite the reader to explore all of the notebooks. All notebooks from section III contains interesting interactive plots. The name of the notebooks corresponds to the section in the report in which the work is described. 

#### Folders:
- SectionB
- SectionC_grid
- SectionD_grid
- SectionE_grid
- SectionF_grid
- mnist_pickles
- mnist
- Data

The first 6 contains pickles of LLE and t-SNE transformations, while the last two contains our datasets. 
