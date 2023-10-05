import qiskit
import numpy as np
import sys
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import FreezeCoreTransformer 
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

import qiskit_nature
qiskit_nature.settings.use_pauli_sum_op = False

from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP, COBYLA, SPSA
from qiskit.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.circuit.library import TwoLocal
from matplotlib import pyplot as plt
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, IBMQ, transpile
from qiskit.circuit import QuantumCircuit, ParameterVector

### UTILITIES ### 
def store_intermediate_data(niter, params, energy, meta):
    """ 
    Store intermediate computed data during the simulation
    """
    computed['energy'].append(energy)
    computed['parameters'].append(params)
    computed['iter'].append(niter)

def get_qubit_hamiltonian(distance):
    """
    Return the second quantization hamiltonian of the H2 molecule, built from
    the given geometry
    """
            
    driver = PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    
    #problem = driver.run()
    problem = FreezeCoreTransformer(
        freeze_core=True, remove_orbitals=[-3, -2]
    ).transform(driver.run())

    second_q_hamiltonian = problem.hamiltonian.second_q_op()

    #mapper = JordanWignerMapper()
    mapper = ParityMapper(num_particles=problem.num_particles)

    return mapper.map(second_q_hamiltonian), problem

def get_ansatz(problem, mapper):
    """
    Return the UCCSD ansatz of the given problem, with the given mapper
    """
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
        ),
    )

    return ansatz


### DOMAIN ###
SHOTS = int(sys.argv[1])
distance = 1.596 
qubit_hamiltonian, problem = get_qubit_hamiltonian(distance)
nuclear_repulsion = problem.hamiltonian.nuclear_repulsion_energy
mapper = ParityMapper(num_particles=problem.num_particles) 
ansatz = get_ansatz(problem, mapper)

#optimizer = SLSQP()
optimizer = COBYLA()

initial_point = [0] * (ansatz.num_parameters)
computed = {
    'energy': [],
    'parameters': [],
    'iter': [],
        }

numpy_solver = NumPyMinimumEigensolver()

### SOLVER ###
vqe_solver = VQE(Estimator(options={'shots': SHOTS}), ansatz, optimizer, callback=store_intermediate_data,
                 initial_point=initial_point)

### RESULTS ###
vqe_solution = vqe_solver.compute_minimum_eigenvalue(qubit_hamiltonian)
exact_solution = numpy_solver.compute_minimum_eigenvalue(qubit_hamiltonian)

vqe_electr_ground_state = vqe_solution.eigenvalue.real
vqe_total_energy = problem.interpret(vqe_solution).total_energies[0].real

exact_elect_ground_state = exact_solution.eigenvalue.real
exact_total_energy = problem.interpret(exact_solution).total_energies[0].real


print(computed['iter'][-1])
print(exact_total_energy)
print(vqe_total_energy)

### PARAMS ###
theta = []
for i in range(ansatz.num_parameters):
    theta.append([])

for params in computed['parameters']:
    for i in range(ansatz.num_parameters):
      theta[i].append(params[i])


### FIDELITY ###
fidelity = []
exact_final_state = exact_solution.eigenstate

for prms in computed['parameters']:
    temp_circuit = ansatz.assign_parameters(prms)
    simulator = Aer.get_backend('statevector_simulator')
    compiled_circuit = transpile(temp_circuit, simulator)
    job = simulator.run(compiled_circuit, shots=SHOTS)
    result = job.result()
    temp_state = result.get_statevector()
    fidelity.append(np.abs(np.dot(temp_state, exact_final_state)))

print(fidelity[-1])


### PLOTS ###
print(computed['iter'][-1])

# ENERGY PLOT #
#shift energies of nuclear_repulsion offset
for i in range(len(computed['energy'])):
    computed['energy'][i] += nuclear_repulsion -7.797875816258
    
plt.plot(computed['iter'], computed['energy'], '.',color='b', label='VQE energy') 
plt.xlabel('Iteration step')
plt.ylabel('Energy(au)')
plt.axhline(y=exact_total_energy, color='r', linestyle='-', label='Exact energy')
plt.legend(loc='upper right')
plt.show()

# FIDELITY PLOT #
plt.plot(computed['iter'], fidelity, '.', color='g')
plt.xlabel('Iteration step')
plt.ylabel('Fidelity')
plt.axhline(y=1, color='r', linestyle='-', label='Exact energy')
plt.show()

# BOND ENERGY PLOT #
vqe_energies = {}
mean_energies = {}
exact_energies = []
distance_range = np.arange(0.4, 4, 0.2)
vqe_bond_dist = -1
prev_vqe_energy = np.Inf
for dist in distance_range:
    vqe_energies[dist] = []
    for i in range(10):
        q_h, prob = get_qubit_hamiltonian(dist)
        ansatz = get_ansatz(prob, mapper)
        initial_point = np.random.random(ansatz.num_parameters)

        vqe_solver = VQE(Estimator(), ansatz, optimizer, callback=store_intermediate_data,
                  initial_point=initial_point)

        vqe_solution = vqe_solver.compute_minimum_eigenvalue(q_h)
        vqe_total_energy = prob.interpret(vqe_solution).total_energies[0].real

        vqe_energies[dist].append(vqe_total_energy)


    exact_solution = numpy_solver.compute_minimum_eigenvalue(q_h)
    exact_total_energy = prob.interpret(exact_solution).total_energies[0].real
    exact_energies.append(exact_total_energy)
    mean_energies[dist] = {'mean': np.average(vqe_energies[dist]), 'std': np.std(vqe_energies[dist])}

res_file = open("statistics/10SPSLQ.txt", "w")
for dist in vqe_energies: 
    stri = f"{dist}"
    res_file.write(stri)
    res_file.write(": [")
    for energy in vqe_energies[dist]:
        stri = f"{energy}, "
        res_file.write(stri)
    res_file.write("]")
    res_file.write(f"mean: {mean_energies[dist]['mean']}, std: {mean_energies[dist]['std']}\n")
res_file.close()
