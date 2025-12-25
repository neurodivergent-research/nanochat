"""
Utilities for injecting specific data at particular steps during training.
"""


def dataloader_with_step_injection(train_loader, inject_at_steps):
    """
    Wrapper generator that intercepts batches from the dataloader and injects
    specific data at certain training steps.

    Args:
        train_loader: The underlying dataloader that yields (inputs, targets, state_dict)
        inject_at_steps: Dict mapping step_number -> (inputs_tensor, targets_tensor)
                        The injected inputs/targets will replace the normal batch at that step.

    Yields:
        (inputs, targets, state_dict) tuples, with inputs/targets potentially replaced
        by injected data at specified steps.

    Example:
        inject_data = {
            100: (special_inputs_1, special_targets_1),
            500: (special_inputs_2, special_targets_2),
        }
        train_loader = dataloader_with_step_injection(train_loader_raw, inject_data)
    """
    step = 0
    for inputs, targets, state_dict in train_loader:
        if step in inject_at_steps:
            inputs, targets = inject_at_steps[step]
        yield inputs, targets, state_dict
        step += 1
