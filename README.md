# Modified Grey Wolf Optimization


## Features
- You can find GWO_modified_v1.py and GWO_modified_v2.py in Optimizers directory.
- GWO_modified_v1 is based on the assumption of center bias of GWO.
- GWO_modified_v2 is based on the assumption of origin bias of GWO.

Run

    pip install -r requirements.txt

(possibly with `sudo`)

This command will install `sklearn`, `NumPy`, `SciPy`, and other dependencies for you.

- **For Windows**: Please install Anaconda from [here](https://www.continuum.io/downloads), which is the leading open data science platform powered by Python.
- **For Ubuntu or Debian (Python 3)**:
  
      sudo apt-get install python3-numpy python3-scipy liblapack-dev libatlas-base-dev libgsl0-dev fftw-dev libglpk-dev libdsdp-dev

## Get the source

Clone the Git repository from GitHub

    git clone https://github.com/7ossam81/EvoloPy.git


## Quick User Guide

EvoloPy toolbox contains twenty three benchmarks (F1-F24). The main file is the optimizer.py, which considered the interface of the toolbox. In the optimizer.py you can setup your experiment by selecting the optimizers, the benchmarks, number of runs, number of iterations, and population size. 
The following is a sample example to use the EvoloPy toolbox.  
Select optimizers from the list of available ones: "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS","HHO","SCA","JAYA","DE". For example:
```
optimizer=["SSA","PSO","GA"]  
```

After that, Select benchmark function from the list of available ones: "F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19". For example:
```
objectivefunc=["F3","F4"]  
```

Select the number of repetitions for each experiment. To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.  For example:
```
NumOfRuns=10  
```
Select general parameters for all optimizers (population size, number of iterations). For example:
```
params = {'PopulationSize' : 30, 'Iterations' : 50}
```
Choose whether to Export the results in different formats. For example:
```
export_flags = {'Export_avg':True, 'Export_details':True, 'Export_convergence':True, 'Export_boxplot':True}
```

Run the example file:
```
python examples/example.py
```

## Useful Links
- **Paper**: 

## Author
Kelly

## Reference

Faris, Hossam, Ibrahim Aljarah, Seyedali Mirjalili, Pedro A. Castillo, and Juan Julián Merelo Guervós. "EvoloPy: An Open-source Nature-inspired Optimization Framework in Python." In IJCCI (ECTA), pp. 171-177. 2016.
https://www.scitepress.org/Papers/2016/60482/60482.pdf

Include the following related citations:

- Qaddoura, Raneem, Hossam Faris, Ibrahim Aljarah, and Pedro A. Castillo. "EvoCluster: An Open-Source Nature-Inspired Optimization Clustering Framework in Python." In International Conference on the Applications of Evolutionary Computation (Part of EvoStar), pp. 20-36. Springer, Cham, 2020.
- Ruba Abu Khurma, Ibrahim Aljarah, Ahmad Sharieh, and Seyedali Mirjalili. Evolopy-fs: An open-source nature-inspired optimization framework in python for feature selection. In Evolutionary Machine Learning Techniques, pages 131–173. Springer, 2020



## Support

Use the [issue tracker](https://github.com/7ossam81/EvoloPy/issues) to report bugs or request features. 


