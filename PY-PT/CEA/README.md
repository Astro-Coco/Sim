# CEA

pyCEA is a easy to install version of rocketCEA. It wraps NASA's CEA code, written in fortran. CEA can simulate combustion performance of propellant mixtures, given chamber conditions.

TO RUN, add the following lines at the top of your code (replace pyCEA_path)
```python
import sys
sys.path.insert(1, 'pyCEA_path')
from pyCEA import cea
```

You can then use
```python
results = cea(inputs)
isp = cea(inputs)['ISP'][-1]
```

by Alexis Angers

