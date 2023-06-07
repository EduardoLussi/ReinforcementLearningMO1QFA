from qiskit import IBMQ, pulse, QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers.fake_provider import FakeBelem
from qiskit_aer import PulseSimulator
import numpy as np
from scipy import spatial

class QFA:
    def __init__(self, p=11, batch_size=1, pulse_duration=64, qubit=0, backend='ibmq_lima'):
        self.p = p                              # MOD^p
        self.pulse_progs = []                   # Pulse programs for the episode
        self.pulse_duration = pulse_duration    # Pulse duration
        self.qubit = qubit                      # Quantum computer qubit
        self.batch_size = batch_size            # Size of repetitions batches

        # Get backend
        # IBMQ.enable_account("43c3552338739471ef9b36f2ad6d17ec2cdfd4885275bead3e1d83d54509666b3f9eff7a8129fa720346d955780b4ebfc3d4f58e16c9ed24f834fad47ac43e84")
        # provider = IBMQ.get_provider("ibm-q")
        # self._backend = provider.get_backend(backend)
        self._backend = PulseSimulator().from_backend(FakeBelem())

        # Expected acceptance probabilities
        self._expected = np.array([np.cos(2*np.pi*w/self.p)**2 for w in range(self.p)])
    
    # Reset environment
    def reset(self):
        self.pulse_progs.clear()
        return np.append(self._expected, 0)

    def step(self, action):
        # Build new pulse program
        with pulse.build(name=f'pulse_prog_{len(self.pulse_progs)}', backend=self._backend) as pulse_prog:
                pulse.play(pulse.library.gaussian_square(
                    duration=self.pulse_duration,
                    sigma=1,
                    amp=action,
                    risefall=1
                ), pulse.drive_channel(self.qubit))
        self.pulse_progs.append(pulse_prog)

        # Build base circuit
        base_circ = QuantumCircuit(1, 1)
        for i, pulse_prog in enumerate(self.pulse_progs):   # For all pulse_programs
            gate_name = f'gate_{i}'
            gate = Gate(gate_name, 1, params=[])

            # Apply pulse batch_size times
            for _ in range(self.batch_size*self.p-1):
                base_circ.append(gate, [self.qubit])
            
            base_circ.add_calibration(gate_name, [self.qubit], pulse_prog)

        # Get measurements for |w| mod p = p..1
        circs = [base_circ]
        for _ in range(self.p-1):
            circ = circs[-1].copy()
            circ.data.pop()
            circs.append(circ)

        for circ in circs:
            circ.measure(0, 0)
        circs = circs[::-1]

        # Execute jobs
        n_shots = 2048
        job = self._backend.run(circs, shots=n_shots)
        job.wait_for_final_state()

        # Get results
        results = job.result()
        
        # Observation is an array of the time and acceptance probabilities
        probabilities = np.array([results.get_counts(i).get('0', 0)/n_shots for i in range(self.p)])
        observation = np.append(probabilities, self.batch_size*self.p*len(self.pulse_progs)-1)
        
        # Reward is e^(-100*(1-similarity))
        reward = 1 - spatial.distance.cosine(probabilities, self._expected)/2
        reward = np.exp(-100*(1-reward))

        # Done if one of the results have an absolute error above 5%
        done = False
        for i, ob in enumerate(probabilities):
            if np.abs(ob - self._expected[i]) > 0.1:
                done = True
                break

        return observation, reward, done



