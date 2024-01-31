# Stochastic Process Simulators (sps)

## Setup
- Install Python 3.11:
    - Install `pyenv`: `curl https://pyenv.run | bash`
    - Copy the lines it says to your `~/.bashrc` and reload `source ~/.bashrc`
    - Install Python 3.11: `pyenv install 3.11`
    - Make Python 3.11 your default: `pyenv global 3.11`
- Install `poetry`: `curl -sSL https://install.python-poetry.org | python3 -`
- Setup env: `cd sps && poetry install`
- Run tests: `poetry run pytest`

## Examples
```python
from sps.gp import GP
from sps.priors import Prior
from sps.utils import build_grid

gp = GP(lengthscale=Prior("beta", {"a": 2.5, "b": 6}))
grid = build_grid([{"start": 0, "stop": 1, "num": 50}] * 2)  # 50x50 grid
var, ls, mu = gp.simulate(grid, batch_size=16, approx=True)  # kronecker
var.shape == (16,)
ls.shape == (16,)
mu.shape == (16, 50, 50, 1)
````
