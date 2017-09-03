import numpy as np


class RandomGenerator(object):
    def __init__(self, generate_space):
        """
        :param generate_space: a dict or a tuple list of (parameter_name, sample log space), los space is a (low, high) tuple
        """
        if isinstance(generate_space, dict):
            generate_space = generate_space.items()
        self.generate_space = generate_space

    def __call__(self, generate_time: int):
        for i in range(generate_time):
            r = {}
            for a, b in self.generate_space:
                r[a] = np.power(10, np.random.uniform(b[0], b[1]))
            yield r


class ConstantGenerator(object):
    def __init__(self, parameter_value):
        """
        :param parameter_value:  a dict or a tuple list of (parameter_name, parameter_value)
        """
        if isinstance(parameter_value, list):
            parameter_value = dict(parameter_value)
        self.parameter_value = parameter_value

    def __call__(self, generate_time: int):
        for i in range(generate_time):
            yield self.parameter_value


class CombineGenerator(object):
    def __init__(self, *args):
        """
        :param args: any generator object
        """
        self.generator = args

    def __call__(self, generate_time: int):
        generator_list = list(map(lambda x: x(generate_time), self.generator))
        for _ in range(generate_time):
            r = {}
            for i in generator_list:
                r.update(next(i))
            yield r


if __name__ == '__main__':
    import copy
    import itertools

    np.random.seed(10)
    sample_param = {"a": (-4, -1), 'b': (1, 6)}
    constant_param = {'c': 'aba', 'd': 2}
    random_generater = RandomGenerator(sample_param)
    costant_generater = ConstantGenerator(constant_param)
    combine_generater = CombineGenerator(random_generater, costant_generater)
    sample_number = 2
    target_sample_param = [{'a': 0.020604492866274665, 'b': 12.698714136216184},
                           {'a': 0.0079605798830767772, 'b': 55465.050493619899}]
    for a, b in zip(random_generater(sample_number), target_sample_param):
        assert a == b, 'random sample error'

    for a, b in zip(costant_generater(sample_number), itertools.repeat(constant_param, times=sample_number)):
        assert a == b, 'constant error'

    target_combine_param = []
    for t in [{'a': 0.0031298320746735273, 'b': 133.04030371488227},
              {'a': 0.00039281548020791377, 'b': 63482.432709463814}]:
        p = copy.deepcopy(constant_param)
        p.update(t)
        target_combine_param.append(p)

    for a, b in zip(combine_generater(sample_number), target_combine_param):
        assert a == b, 'combine error'
