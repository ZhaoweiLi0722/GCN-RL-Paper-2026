"""P2 analytical qualification and same-start forks; reuse serial P1 mechanics."""

import numpy as np
import torch

from src.models.candidate_policy import CandidatePolicy
from src.models.reference_prior_candidate import ReferencePriorCandidatePolicy
from src.rl.candidate_imitation import decode_example
from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_campaign import PilotCampaign
from src.rl.candidate_pilot_driver import fork_policy
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_rollout import evaluate_candidate_policy
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.routing_candidate_contract import choose_candidate


def qualify_prior(kernel, examples, *, expected_rows):
    if type(kernel.policy) is not ReferencePriorCandidatePolicy or kernel.mode != "frozen":
        raise ValueError("untrained explicit reference-prior owner required")
    if len(examples) != expected_rows or not examples or len({e["identity"] for e in examples}) != len(examples):
        raise ValueError("exact independent prior qualification rows required")
    before = state_digest(kernel.state_dict())
    rows, inputs = [], []
    for example in examples:
        if example["split"] != "qualification":
            raise ValueError("prior qualification cannot use training/test rows")
        obs, bank = decode_example(example, kernel.contract)
        evaluation = evaluate_candidate_policy(kernel.policy, obs, bank, kernel.contract)
        inputs.append(evaluation.actor_state)
        with torch.no_grad():
            residual = CandidatePolicy.forward(kernel.policy, obs, bank)
        k = len(bank.class_keys)
        expected = np.full(k, kernel.policy.nonreference_mass / max(k - 1, 1))
        expected[bank.reference_class] = 1 - kernel.policy.nonreference_mass if k > 1 else 1.
        probabilities = np.exp(evaluation.log_probs)
        error = float(np.max(np.abs(probabilities - expected)))
        choice = choose_candidate(bank, int(np.argmax(evaluation.log_probs)))
        passed = (choice.submitted_request == bank.requests[0] and error <= 8 * np.finfo(np.float32).eps
                  and bool((probabilities > 0).all()) and evaluation.value == 0.
                  and bool(torch.equal(residual.logits, torch.zeros_like(residual.logits)))
                  and residual.value.item() == 0. and residual.actor_state == evaluation.actor_state)
        rows.append({"identity": example["identity"], "classes": k, "reference_class": bank.reference_class,
                     "probabilities": probabilities.tolist(), "max_probability_error": error,
                     "exact_original_reference_request": choice.submitted_request == bank.requests[0], "passed": passed})
    if state_digest(kernel.state_dict()) != before:
        raise ValueError("prior qualification mutated weights or RNG")
    return {"passed": all(r["passed"] for r in rows), "rows": rows, "row_count": len(rows),
            "singleton_rows": sum(r["classes"] == 1 for r in rows), "kernel_sha256": before,
            "public_actor_inputs_sha256": state_digest(inputs),
            "optimizer_steps": 0, "sampling_equivalent_to_deterministic_reference": False}


class ReferencePriorCampaign(PilotCampaign):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.get("pilot_profile") != "p2_reference_prior":
            raise ValueError("explicit P2 profile required")

    def initialize(self, key):
        raise ValueError("P2 forbids BC initialization")

    def collect_shared(self, split):
        if split != "qualification":
            raise ValueError("P2 has no demonstration phase")
        return super().collect_shared(split)

    def qualify_and_fork(self):
        evidence = self.collect_shared("qualification")
        expected = self.config["objective"]["horizon"] * self.config["initialization"]["qualification_episodes_per_block"]
        for key, template in self.templates.items():
            block = int(key.split("/")[0][5:])
            self.qualified[key] = qualify_prior(template, self.qualification[block], expected_rows=expected)
        input_parity = all(len({self.qualified[f"block{b}/{r['name']}"].get("public_actor_inputs_sha256")
                              for r in self.config["representations"]}) == 1 for b in self.config["blocks"])
        path = self.root / "payload" / "qualification.json"
        write_json_once(path, self.qualified)
        if not input_parity or not all(r["passed"] for r in self.qualified.values()):
            raise ValueError("analytical reference-prior qualification failed; attempt closed")
        for key, template in self.templates.items():
            if self.qualified[key]["kernel_sha256"] != state_digest(template.state_dict()):
                raise ValueError("qualified prior changed before fork")
            b, rep = key.split("/")
            for role, owner in fork_policy(template, self.config, self.streams,
                    block=int(b[5:]), representation=rep).items():
                name = f"{key}/{role}"
                self.models[name] = owner
                initial = self.root / "payload" / "models" / name / "initial.pt"
                save_envelope(initial, owner.state_dict())
                evidence.append(initial)
                if role == "frozen":
                    self.model_paths[name] = initial
        return evidence + [path]
