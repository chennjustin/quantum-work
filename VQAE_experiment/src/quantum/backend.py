"""Backend factory for ideal and IBM fake noisy simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorSampler

BackendMode = Literal["ideal", "ibm_fake_noisy"]

FAKE_BACKEND_REGISTRY: dict[str, str] = {
    "fake_manila": "FakeManilaV2",
    "fake_sherbrooke": "FakeSherbrooke",
    "fake_belem": "FakeBelemV2",
    "fake_quito": "FakeQuitoV2",
    "fake_lima": "FakeLimaV2",
}


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend cannot be created."""


@dataclass
class CircuitRunMetadata:
    """Metadata captured for each backend execution."""

    backend_mode: str
    fake_backend_name: str | None
    shots: int
    seed: int
    optimization_level: int
    transpiled_depth: int | None = None
    two_qubit_gate_count: int | None = None
    total_gate_count: int | None = None


@dataclass
class BackendRunner:
    """Execute one-bit measurement circuits on ideal or noisy backends."""

    mode: BackendMode
    shots: int
    seed: int
    optimization_level: int = 1
    fake_backend_name: str | None = None
    _simulator: object | None = field(default=None, repr=False)
    _ideal_sampler: StatevectorSampler | None = field(default=None, repr=False)
    last_metadata: CircuitRunMetadata | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.mode == "ideal":
            self._ideal_sampler = StatevectorSampler(seed=self.seed)
            return

        if self.mode != "ibm_fake_noisy":
            raise ValueError(f"Unknown backend mode: {self.mode}")

        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise BackendUnavailableError(
                "qiskit-aer is required for ibm_fake_noisy mode. "
                "Install with: python -m pip install 'qiskit-aer~=0.17'"
            ) from exc

        if not self.fake_backend_name:
            raise ValueError("fake_backend_name is required for ibm_fake_noisy mode")

        class_name = FAKE_BACKEND_REGISTRY.get(self.fake_backend_name)
        if class_name is None:
            known = ", ".join(sorted(FAKE_BACKEND_REGISTRY))
            raise ValueError(f"Unknown fake backend '{self.fake_backend_name}'. Known: {known}")

        try:
            import qiskit_ibm_runtime.fake_provider as fake_provider
            fake_backend_cls = getattr(fake_provider, class_name)
        except AttributeError as exc:
            raise BackendUnavailableError(
                f"Fake backend class {class_name} not found in qiskit_ibm_runtime.fake_provider"
            ) from exc

        fake_backend = fake_backend_cls()
        self._simulator = AerSimulator.from_backend(fake_backend)

    @property
    def uses_statevector_sampler(self) -> bool:
        return self.mode == "ideal"

    def required_qubits(self, circuit: QuantumCircuit) -> int:
        return circuit.num_qubits

    def ensure_capacity(self, circuit: QuantumCircuit) -> None:
        """Raise if the circuit exceeds backend qubit capacity."""

        needed = self.required_qubits(circuit)
        if self.mode == "ibm_fake_noisy":
            assert self._simulator is not None
            capacity = self._simulator.configuration().n_qubits
            if needed > capacity:
                raise ValueError(
                    f"Circuit requires {needed} qubits but backend "
                    f"'{self.fake_backend_name}' supports only {capacity}"
                )

    def run_one_bit_measurement(self, circuit: QuantumCircuit) -> float:
        """Return P(measured classical bit = 1)."""

        self.ensure_capacity(circuit)
        if self.mode == "ideal":
            return self._run_ideal(circuit)
        return self._run_noisy(circuit)

    def run_batch_one_bit_measurements(self, circuits: list[QuantumCircuit]) -> list[float]:
        """Return P(bit=1) for each circuit."""

        if not circuits:
            return []
        for circuit in circuits:
            self.ensure_capacity(circuit)

        if self.mode == "ideal":
            assert self._ideal_sampler is not None
            result = self._ideal_sampler.run(circuits, shots=self.shots).result()
            probs = []
            for pub in result:
                counts = pub.data.c.get_counts()
                probs.append(float(counts.get("1", 0) / self.shots))
            self.last_metadata = CircuitRunMetadata(
                backend_mode=self.mode,
                fake_backend_name=self.fake_backend_name,
                shots=self.shots,
                seed=self.seed,
                optimization_level=self.optimization_level,
            )
            return probs

        return [self._run_noisy(c) for c in circuits]

    def _run_ideal(self, circuit: QuantumCircuit) -> float:
        assert self._ideal_sampler is not None
        result = self._ideal_sampler.run([circuit], shots=self.shots).result()
        counts = result[0].data.c.get_counts()
        self.last_metadata = CircuitRunMetadata(
            backend_mode=self.mode,
            fake_backend_name=None,
            shots=self.shots,
            seed=self.seed,
            optimization_level=self.optimization_level,
        )
        return float(counts.get("1", 0) / self.shots)

    def _run_noisy(self, circuit: QuantumCircuit) -> float:
        assert self._simulator is not None
        transpiled = transpile(
            circuit,
            backend=self._simulator,
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed,
        )
        job = self._simulator.run(transpiled, shots=self.shots, seed_simulator=self.seed)
        result = job.result()
        counts = result.get_counts(transpiled)
        prob_one = float(counts.get("1", 0) / self.shots)

        two_q = sum(1 for inst in transpiled.data if inst.operation.name == "cx")
        self.last_metadata = CircuitRunMetadata(
            backend_mode=self.mode,
            fake_backend_name=self.fake_backend_name,
            shots=self.shots,
            seed=self.seed,
            optimization_level=self.optimization_level,
            transpiled_depth=transpiled.depth(),
            two_qubit_gate_count=two_q,
            total_gate_count=len(transpiled.data),
        )
        return prob_one


def create_backend_runner(
    mode: BackendMode,
    fake_backend_name: str = "fake_manila",
    shots: int = 4096,
    seed_simulator: int = 123,
    optimization_level: int = 1,
) -> BackendRunner:
    """Create a backend runner. Never silently falls back to ideal simulation."""

    if mode == "ideal":
        return BackendRunner(
            mode="ideal",
            shots=shots,
            seed=seed_simulator,
            optimization_level=optimization_level,
        )

    if mode == "ibm_fake_noisy":
        return BackendRunner(
            mode="ibm_fake_noisy",
            shots=shots,
            seed=seed_simulator,
            optimization_level=optimization_level,
            fake_backend_name=fake_backend_name,
        )

    raise ValueError(f"Unsupported backend mode: {mode}")
