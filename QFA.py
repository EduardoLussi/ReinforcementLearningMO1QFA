from qiskit import IBMQ, pulse, QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers.fake_provider import FakeBelem
from qiskit_aer import PulseSimulator
import numpy as np

class QFA:
    def __init__(self, p=11, batch_size=2, pulse_duration=64, qubit=0, backend='ibmq_lima'):
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
        self._expected = np.cos(2*np.pi*0/self.p)**2
    
    # Reset environment
    def reset(self):
        self.pulse_progs.clear()
        return np.array([self._expected, 0])

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
            for _ in range(self.batch_size*self.p):
                base_circ.append(gate, [self.qubit])
            
            base_circ.add_calibration(gate_name, [self.qubit], pulse_prog)
        base_circ.measure(0, 0)

        # Execute jobs
        n_shots = 2048
        job = self._backend.run(base_circ, shots=n_shots)
        job.wait_for_final_state()

        # Get result
        result = job.result()
        
        # Observation is an array of the time and acceptance probabilities
        probability = result.get_counts().get('0', 0)/n_shots
        repetitions = self.batch_size*self.p*len(self.pulse_progs)
        observation = np.array([probability, repetitions/1000])

        # Done if one of the results have an absolute error above 5%
        absolute_error = np.abs(probability - self._expected)
        # Reward is exponential decay
        reward = np.exp(-10*(1 - (1-absolute_error)))
        if absolute_error > 0.05:
            if absolute_error > 0.5:
                reward = 0
            else:
                reward /= 2
            done = True
        elif repetitions >= 374:
            done = True
        else:
            done = False

        return observation, reward, done



