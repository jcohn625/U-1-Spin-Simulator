# U-1-Spin-Simulator
Statevector simulation with U(1) symmetry
<<<<<<< Updated upstream



=======
```python
import numpy as np
from stavevec_utils import *

N =24 # number of qubits
K = 12 # number of zero bits/ Magnetization sector Hilbert space is N choose K

basis =  build_basis_u64(N,K) # Build the basis states
cache = SwapCache(basis) # Initialize gate cache

psi = init_singlet_product_state(basis,N) # creates product state (|01> - |10>) ^{N/2}
psi_0 = psi.copy()

apply_gate(cache, psi, 1, 2,np.pi/16) # apply e^{-i*(\pi/16) *S_{12}}|\psi>
````
>>>>>>> Stashed changes
