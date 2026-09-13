"""Small atomic mutation helpers for validated retained values."""


def validated_setattr(obj, name, value):
    """Validate a candidate on a shallow copy before touching live state."""
    import copy
    if getattr(obj, "_initialized", False) and not name.startswith("_"):
        candidate = copy.copy(obj)
        object.__setattr__(candidate, name, value)
        candidate._validate()
    object.__setattr__(obj, name, value)


def atomic_transforms(method):
    """Rollback compound transform operations, preserving Transform identity."""
    from functools import wraps
    from inspect import signature

    first_parameter = next(iter(signature(method).parameters))

    @wraps(method)
    def wrapped(*args, **kwargs):
        target = args[0] if args else kwargs.get(first_parameter)
        if hasattr(target, "_get_transform_roots"):
            objects = target._get_transform_roots()
        elif isinstance(target, (list, tuple)):
            objects = target
        else:
            objects = [target]
        states = [(obj.transform, obj.transform.__dict__.copy())
                  for obj in objects if hasattr(obj, "transform")]
        try:
            return method(*args, **kwargs)
        except Exception:
            for transform, state in states:
                transform.__dict__.clear()
                transform.__dict__.update(state)
            raise
    return wrapped
