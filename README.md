# coptim: A Convex Optimization Package

coptim is a Python library for solving convex smooth/non-smooth objective functions.

## Documentation

Sphinx Documentation is available [here](https://cmazzaanthony.github.io/coptim/).

## Installation

```bash
git clone git@github.com:cmazzaanthony/coptim.git
cd coptim
pip install -e .
```

## Usage

```python
from coptim.optimizers.gradient_method import GradientMethod
from coptim.functions.rosenbrock import Rosenbrock

objective = Rosenbrock()
starting_point = np.array([-1.2, 1])
beta = 0.5
sigma = 0.0001
epsilon = 0.0001

optimizer = GradientMethod()

x = optimizer.optimize(starting_point,
                       objective,
                       beta,
                       sigma,
                       epsilon)

print(f'Optimal Point: {x}')
print(f'Iterations: {optimizer.iterations}')
```

```
Optimal Point: [0.99992058 0.9998407 ]
Iterations: 8058
```

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README.
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
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── functions      <- Python Files for Mathematical Functions
        │
        ├── optimizers     <- Scripts to turn raw data into features for modeling
        │
        ├── function.py    <- Abstract class for implementing functions
        ├── optimizer.py   <- Abstract class for optimizer

--------

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
