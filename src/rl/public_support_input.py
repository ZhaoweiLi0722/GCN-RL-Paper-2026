"""Full, same-information public input for the unregistered v2 capacity pilot."""

from dataclasses import dataclass

from src.env.patient_support_public import PublicSupportOperations, validate_public_operations
from src.rl.public_support_collector import PublicSupportInput


@dataclass(frozen=True)
class PublicSupportControlInput:
    common: PublicSupportInput
    operations: PublicSupportOperations

    @classmethod
    def capture(cls, host):
        common = PublicSupportInput.capture(host)
        operations = validate_public_operations(host.public_operations())
        if operations.epoch != common.epoch or operations.site_ids != common.site_ids:
            raise ValueError("public patient/resource view does not share the control boundary")
        by_id = {row.patient_id: row for row in operations.patients}
        ready = tuple(sum(by_id[pid].support_complete for pid in queue) for queue in operations.waiting_order)
        if ready != common.ready_waiting_counts:
            raise ValueError("public ready counts and patient identities disagree")
        if operations.last_service is not None:
            event = operations.last_service
            latest = tuple(site[-1] for site in common.capacity_history)
            for i, row in enumerate(latest):
                if (row[1] != event.applied_hours[i] or row[2] != event.ordinary_hours[i]
                        or row[3] != len(event.eligible_order[i]) or row[4] != len(event.completed_ids[i])
                        or row[5] != 1.0):
                    raise ValueError("public aggregate and identity service receipts disagree")
        return cls(common, operations)
