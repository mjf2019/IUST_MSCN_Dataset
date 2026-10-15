"""Reward contract for the optional three-action routing experiment."""
import math


def quality(prediction, label):
    return 1.0 if int(prediction) == int(label) else -1.0


def routing_reward(action, frozen_prediction, context_prediction, label,
                   loss_before=None, loss_after=None, selection_cost=0.2,
                   gain_weight=1.0, epsilon=1e-8):
    if action == 0:
        return quality(frozen_prediction, label)
    if action == 1:
        return quality(context_prediction, label)
    if action != 2:
        raise ValueError("Action must be 0, 1, or 2.")
    if loss_before is None or loss_after is None:
        raise ValueError("Action 2 requires CE before and after the update.")
    if any(not math.isfinite(v) or v < 0 for v in
           (loss_before, loss_after, selection_cost, gain_weight)):
        raise ValueError("Losses, cost and gain weight must be finite and nonnegative.")
    gain = (loss_before - loss_after) / (loss_before + loss_after + epsilon)
    return quality(context_prediction, label) + gain_weight * gain - selection_cost


def prediction_route(action):
    if action not in (0, 1, 2):
        raise ValueError("Invalid action.")
    return "frozen" if action == 0 else "context"


def available_actions(query_allowed):
    return (0, 1, 2) if query_allowed else (0, 1)
