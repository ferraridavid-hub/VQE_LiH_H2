import qiskit 
import sys
import numpy as np
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

import qiskit_nature
qiskit_nature.settings.use_pauli_sum_op = False

from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP, COBYLA, SPSA
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit.circuit.library import TwoLocal
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.circuit.library import TwoLocal
from matplotlib import pyplot as plt
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, IBMQ
from qiskit.circuit import QuantumCircuit, ParameterVector

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService

# COMMENTARE PER LAVORARE IN CLOUD
from qiskit.primitives import Estimator

# DECOMMENTARE PER LAVORARE IN CLOUD
#from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Options



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
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    
    problem = driver.run()
    second_q_hamiltonian = problem.hamiltonian.second_q_op()

    #mapper = JordanWignerMapper()
    mapper = ParityMapper(problem.num_particles)

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
ITER = 1
distance = 0.735
_, problem = get_qubit_hamiltonian(distance)
mapper = ParityMapper(problem.num_particles)
optimizer = COBYLA()
computed = {
    'energy': [],
    'parameters': [],
    'iter': [],
        }
numpy_solver = NumPyMinimumEigensolver()

route = sys.argv[2]
if route == 'bond': 
    print('bond')
elif route == 'single':
    print('single')
else:
    print('usage: h2.real shots <bond|single>')
    sys.exit(1)

if route == 'single':
    distance = 0.735
    qubit_hamiltonian, problem = get_qubit_hamiltonian(distance)
    nuclear_repulsion = problem.hamiltonian.nuclear_repulsion_energy
    print(nuclear_repulsion)
    mapper = ParityMapper(problem.num_particles)
    ansatz = get_ansatz(problem, mapper)
    
    initial_point = [0] * (ansatz.num_parameters)
    computed = {
        'energy': [],
        'parameters': [],
        'iter': [],
            }
    
    ### SOLVER ###
    
    # DECOMMENTARE PER LAVORARE IN CLOUD E INDENTARE
    
    #provider=IBMProvider()
    #service = QiskitRuntimeService()

    # Select a backend.
    #backend = service.least_busy(min_num_qubits=2, simulator=False) 
    
    #print(backend)
    
    #with Session(service=service, backend=backend.name) as session:
    #    estimator = Estimator(session=session, options={"shots": SHOTS})
    #    vqe_solver = VQE(estimator, ansatz, optimizer, callback=store_intermediate_data, initial_point=initial_point)
    
    # COMMENTARE PER LAVORARE IN CLOUD
    estimator = Estimator(options={'shots':SHOTS})
    vqe_solver = VQE(estimator, ansatz, optimizer, callback=store_intermediate_data, initial_point=initial_point)
    
    ### RESULTS ###
    calc = GroundStateEigensolver(mapper, vqe_solver)
    res = calc.solve(problem)
    
    mean_fid = []
    mean_energy = []
    for i in range(ITER):
        vqe_solver = VQE(estimator, ansatz, optimizer, callback=store_intermediate_data, initial_point=initial_point)
        computed = {
            'energy': [],
            'parameters': [],
            'iter': [],
                }
        vqe_solution = vqe_solver.compute_minimum_eigenvalue(qubit_hamiltonian)
        exact_solution = numpy_solver.compute_minimum_eigenvalue(qubit_hamiltonian)
        
        vqe_electr_ground_state = vqe_solution.eigenvalue.real
        vqe_total_energy = problem.interpret(vqe_solution).total_energies[0].real
        
        exact_elect_ground_state = exact_solution.eigenvalue.real
        exact_total_energy = problem.interpret(exact_solution).total_energies[0].real
        
        mean_energy.append(vqe_total_energy)
        ### FIDELITY ###
        fidelity = []
        vqe_states = []
        exact_final_state = exact_solution.eigenstate
        
        for prms in computed['parameters']:
            temp_circuit = ansatz.assign_parameters(prms)
            simulator = Aer.get_backend('statevector_simulator')
            compiled_circuit = transpile(temp_circuit, simulator)
            job = simulator.run(compiled_circuit, shots=SHOTS)
            result = job.result()
            temp_state = result.get_statevector()
            vqe_states.append(temp_state)
            fidelity.append(np.abs(np.dot(temp_state, exact_final_state)))
        mean_fid.append(fidelity[-1])
    
    print(f'mean energy: {np.average(mean_energy)} +- {np.std(mean_energy)}')
    print(f'mean fidelity: {np.average(mean_fid)} +- {np.std(mean_fid)}')
    
    ### PLOTS ###
    
    # ENERGY PLOT #
    #shift energies of nuclear_repulsion offset
    for i in range(len(computed['energy'])):
        computed['energy'][i] += nuclear_repulsion + 0.03
        
    plt.plot(computed['iter'], computed['energy'], marker='o',markersize=4, linestyle='None', label='VQE energy') 
    plt.xlabel('Iteration step')
    plt.ylabel('Energy(au)')
    plt.axhline(y=exact_total_energy, color='orange', linestyle='-', label='Exact energy')
    plt.legend(loc='upper right')
    plt.show()
    
    # FIDELITY PLOT #
    plt.plot(computed['iter'], fidelity, marker='o', markersize=4, linestyle='None',color='g')
    plt.xlabel('Iteration step')
    plt.ylabel('Fidelity')
    plt.axhline(y=1, color='r', linestyle='-', label='Exact energy')
    plt.show()
   # session.close()
elif route == 'bond':
    provider=IBMProvider()
    service = QiskitRuntimeService()

    # Select a backend.
    backend = service.least_busy(min_num_qubits=2, simulator=False) 
    print(backend)
    ## BOND ENERGY PLOT #
    optimizer = COBYLA()

    vqe_energies = {}
    mean_energies = {}
    exact_energies = {}
    distance_range = np.arange(0.335, 4.335, 0.2)
    
    # DECOMMENTARE PER LAVORARE IN CLOUD
    with Session(service=service, backend=backend.name) as session:
        for dist in distance_range:
            vqe_energies[dist] = []
            q_h, prob = get_qubit_hamiltonian(dist)
            ansatz = get_ansatz(prob, mapper)
            initial_point = [0] * (ansatz.num_parameters)
            for i in range(ITER):
        
                estimator = Estimator(session=session, options={'shots':SHOTS})
                vqe_solver = VQE(estimator, ansatz, optimizer, callback=store_intermediate_data, initial_point=initial_point)
        
                vqe_solution = vqe_solver.compute_minimum_eigenvalue(q_h)
                vqe_total_energy = prob.interpret(vqe_solution).total_energies[0].real
    
                vqe_energies[dist].append(vqe_total_energy)
                print(vqe_total_energy)
        
            exact_solution = numpy_solver.compute_minimum_eigenvalue(q_h)
            exact_energies[dist] = prob.interpret(exact_solution).total_energies[0].real
            mean_energies[dist] = {'mean': np.average(vqe_energies[dist]), 'std': np.std(vqe_energies[dist])}
        
    session.close()
    
    # COMMENTARE PER LAVORARE IN CLOUD
    #for dist in distance_range:
    #    vqe_energies[dist] = []
    #    q_h, prob = get_qubit_hamiltonian(dist)
    #    ansatz = get_ansatz(prob, mapper)
    #    initial_point = [0] * (ansatz.num_parameters)
    #    for i in range(ITER):
    #        estimator = Estimator(options={'shots':SHOTS})
    #        vqe_solver = VQE(estimator, ansatz, optimizer, callback=store_intermediate_data, initial_point=initial_point)
    #
    #        vqe_solution = vqe_solver.compute_minimum_eigenvalue(q_h)
    #        vqe_total_energy = prob.interpret(vqe_solution).total_energies[0].real
    #
    #        vqe_energies[dist].append(vqe_total_energy)
    #
    #    exact_solution = numpy_solver.compute_minimum_eigenvalue(q_h)
    #    exact_energies[dist] = prob.interpret(exact_solution).total_energies[0].real
    #    mean_energies[dist] = {'mean': np.average(vqe_energies[dist]), 'std': np.std(vqe_energies[dist])}
    
    res_file = open(f"statistics/{SHOTS}/h2_energies.txt", "w")
    for dist in vqe_energies: 
        res_file.write(f"{dist}: [")
        for energy in vqe_energies[dist]:
            stri = f"{energy}, "
            res_file.write(stri)
        res_file.write("]")
        res_file.write(f"mean: {mean_energies[dist]['mean']}, std: {mean_energies[dist]['std']}\n")
    res_file.close()
    
    res_file = open(f"statistics/{SHOTS}/h2_mean_std.txt", "w")
    for dist in vqe_energies: 
            res_file.write(f"{dist}: {mean_energies[dist]['mean']}, {mean_energies[dist]['std']}, \n")
    res_file.close()
    
    res_file = open("statistics/h2_exact_energies.txt", "w")
    for dist in exact_energies:
            res_file.write(f"{dist}: {exact_energies[dist]}, \n")
    res_file.close()
