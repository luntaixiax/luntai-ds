
def check_params(param, allowed_types: list = None, allowed_values: list = None):
    if allowed_types:
        if not isinstance(param, tuple(allowed_types)):
            raise TypeError(f"types allowed for {param}: {','.join(allowed_types)}")

    if allowed_values:
        if param not in allowed_values:
            raise ValueError(f"allowed values for {param}: {','.join(allowed_values)}")

    return param