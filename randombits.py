import gmpy
import numpy as np
import scipy.special


def inverse_entropy(H):
    """
    A utility method the returns the corresnponding probability for a given binary entropy.
    """
    if np.isclose(H, 1.0):
        return 0.5    
    p = 0.5*(1.0+np.sqrt(1.0-H))  # H(p) ~ 4.0p(1-p)    
    f = H+p*np.log2(p)+(1-p)*np.log2(1-p)    
    ft = np.log2(p/(1.0-p))
    return p - f/ft # Newton's iteration


def precalculate_probabilities(p, bits=8):
    """
    Pre-calculates the probability of drawing each of the 2^bits patterns,
    so that the probability that any bit will be 1, is p.
    Returns the patterns and the corresponding stochastic vector.
    """

    weights = np.array([scipy.special.binom(bits, i)*(p**i)*((1-p)**(bits-i)) for i in xrange(bits+1)])
    values = np.arange(2**bits, dtype=np.ubyte)
    probabilities = np.zeros(values.shape)
    for i, v in enumerate(values):
        probabilities[i] = weights[gmpy.popcount(i)]/(scipy.special.binom(bits, gmpy.popcount(i))+0.0)
    return values, probabilities


class RandomBitstreams(object):
    """
    This class is used to generate sequences of pseudo-random Bernoulli trials with probability p. 
    """

    def __init__(self, p, chunk_bytes=1, seed=None):
        if seed:        
            np.random.seed(seed)
        self._chunk_bytes = chunk_bytes
        self._values, self._probabilities = precalculate_probabilities(p, chunk_bytes*8)
        
    def bitstream(self, size_in_bytes, seed=None):
        if seed:        
            np.random.seed(seed)
        return np.random.choice(self._values, size=size_in_bytes/self._chunk_bytes, p=self._probabilities)

    
def xor_adjoint_probabilities(p):
    """
    Returns a pair of probabilities, used to generate pseudo-random blocks
    by xoring, whose target probability is p.
    """

    calc = lambda p: ((1-p)/2.0, (1.5*p-0.5)/p)
    if p >= 0.5:
        return calc(p)
    else:
        x, y = calc(1-p)
        return x, 1-y


class CombinatorialGenerator(object):
    """
    This class maintains 2 pools of "building blocks", used to generate new pseudo-random bloks by xoring. 
    """

    def __init__(self, p, block_bytes, pool_size):
        pH, qH = xor_adjoint_probabilities(p)

        gen_p = RandomBitstreams(pH)
        gen_q = RandomBitstreams(qH)

        components = int(np.ceil(np.sqrt(pool_size)))
        self._ps = [gen_p.bitstream(block_bytes) for i in xrange(components)]
        self._qs = [gen_q.bitstream(block_bytes) for i in xrange(components)]
        self._block_bytes = block_bytes

    def components(self):
        return len(self._ps)

    def block_size(self):
        return self._block_bytes

    def block(self, i, j):
        return np.bitwise_xor(self._ps[i], self._qs[j])


def combine(bytes_count, combiner):
    """
    A utility method for piecewise generation of long psuedo-random streams,
    using an instance of CombinatorialGeneraotr.
    """

    k = combiner.components()
    block_bytes = combiner.block_size()
    count = int(np.ceil(bytes_count/(block_bytes+0.0)))   
    return np.hstack([combiner.block(np.random.randint(k), np.random.randint(k)) for i in xrange(count)])[:bytes_count]
