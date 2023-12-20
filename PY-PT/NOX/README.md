# NOX

Nitrous oxide properties and injection simulation.

NOTE: SI UNITS are used in this folder, including pressure in Pa.

## n2o.py

Returns thermophysical properties of nitrous oxide at given temperature / pressure. To use it, add:
```python
import sys
sys.path.insert(1, 'pyNOX_path')
import n2o
```

You can then use the n2o library in your code like so:
```python
density = n2o.rhoLiqSat(temp)
```

## inj.py

Simulation of nitrous oxide bi-phase injection. To use it, add:
```python
import sys
sys.path.insert(1, 'pyNOX_path')
from inj import inject
```

You can then use the inject function in your code like so:
```python
G, P, T, x = inject(P_in, x_in)
```
G is the oxidant mass flux, P is the exit pressure, T is the exit temperature and x the exit vapor volume fraction. You can also run from command line, using:
```cmd
python inj.py -c <Cd> -d <diameter> -p <pressure> -u <d_units>-<p_units>
```

Units are optional, SI is assumed. Per example:
```cmd
python inj.py -c 0.7 -d 0.5 -p 600 -u in-psi
```

## inj-map.py

Allows user to fit a polynomial surface on injection data, for speed.

## tank.py

Nitrous oxide tank emptying simulation. To use it, add:
```python
import sys
sys.path.insert(1, 'pyNOX_path')
import tank
```



By Alexis Angers