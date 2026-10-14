"""Actor-only beta feedback using the frozen v57 normal references."""
from scripts.v52_agent import ContinuationAgent
from scripts.v54_window_readout import SignedWindowReadout
from scripts.v57_phase_readout import PhaseDepartureReadout

VARIANTS = ("fixed-minus2", "fixed-minus2p3", "pooled-beta", "phase-beta")


class OnlinePhaseReadout(SignedWindowReadout):
    def __init__(self, model, pooled, phase):
        super().__init__(model, pooled)
        self.departure = PhaseDepartureReadout(phase)


class ClosedLoopAgent(ContinuationAgent):
    def __init__(self, env, variant, device="cuda"):
        super().__init__(env, variant, device)
        self.readout_coefficient = 0.

    @property
    def penalty_coefficient(self):
        if self.variant == "fixed-minus2":
            return 0.
        if self.variant == "fixed-minus2p3":
            return .3
        return self.readout_coefficient

    def start_shared(self, checkpoint, shared, model, pooled, phase):
        super().start_shared(checkpoint, shared)
        self.readouts = (SignedWindowReadout(model, pooled) if self.variant == "pooled-beta"
                         else OnlinePhaseReadout(model, pooled, phase))
        self.readout_coefficient = 0.

    def observe_transition(self, state, executed_action, next_state, episode_age):
        self.readout_coefficient = float(self.readouts.observe(state, executed_action, next_state, episode_age))
