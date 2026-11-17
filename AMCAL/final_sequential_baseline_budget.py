"""Shared per-domain budget boundary for persistent baseline learners."""


def reset_domain_budget(learner, native):
    """Keep fitted models, replay, thresholds, caches, detectors and RNG."""
    if native == 'ral':
        if getattr(learner.model,'_online_update_window',0):
            raise ValueError('Sequential RAL comparison requires the full-refit protocol.')
        learner.model._n_seen = 0
        learner.model._n_acquired_samples = 0
        learner.model._budget = .2
    elif native in ('calmid','micfoal'):
        if learner.model.pending is not None:
            raise ValueError('Unresolved native feedback at domain boundary.')
        learner.model.seen = 0
        learner.model.queried = 0
    elif native not in ('dfe','arf'):
        raise ValueError('Unknown persistent native baseline.')
