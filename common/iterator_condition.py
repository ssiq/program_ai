

class Condition:
    def __init__(self, condition_fn):
        if not condition_fn:
            condition_fn = self.default_condition
        self.condition_fn = condition_fn
        self.modify = False

    def __call__(self, *args, **kwargs):
        return self.condition_fn(*args, **kwargs)

    def default_condition(self, *args, **kwargs):
        return True

    def is_modify(self):
        return self.modify

