import sys
import argparse
import itertools

import numpy as np
import scipy.interpolate

from randombits import inverse_entropy, CombinatorialGenerator


def plan_blocks(unq_count, dup_count, duplicity):    
    """
    Given the number of required unique blocks (unq_count), the number of required
    duplicated blocks (dup_count) and the duplication factor (duplicty), returns 
    parameters for an appropriate CombinatorialBlocks instance, and a list of 
    indices pairs for the blocks, sorted according to the final layout.
    """

    components = int(np.ceil(np.sqrt(unq_count+dup_count)))
    references = list(itertools.product(xrange(components), xrange(components)))
    count = dup_count*duplicity + unq_count
    return np.random.permutation(np.array(references[:dup_count]*duplicity + references[dup_count:])), components, count


def generate(filename, block_size_kb, unq_count, dup_count, duplicity, entropy):
    """
    This method implements the main logic: Create a new file (filename) that contains
    a random sequence of blocks (each of size block_size_kb) kilobytes. The length of
    the sequence is (unq_count + dup_count*duplicty). Each block has the specified entropy.
    """

    def progress(i, pos): 
        sys.stdout.write('\rProgress: {0:.2f}MB [{1}] {2}%'.format(i*block_size_kb/1024.0, '#'*pos, pos))
        sys.stdout.flush()

    plan, components, count = plan_blocks(unq_count, dup_count, duplicity)
    combiner = CombinatorialGenerator(inverse_entropy(entropy), block_size_kb*1024, unq_count+dup_count)
    progress_unit = int(np.ceil(count/100.0))
    with open(filename, 'wb') as f:
        buff = []
    	for i, (bidA, bidB) in enumerate(plan[:count]):
            if i%progress_unit == 0:
                progress(i, int(np.ceil(i/(progress_unit+0.0))))
            buff.append(combiner.block(bidA, bidB))
            if len(buff)*block_size_kb >= 1024*10:
                f.write(np.hstack(buff).tobytes())
                buff = []
        if len(buff) > 0:
            f.write(np.hstack(buff).tobytes())
            buff = []
    progress(count, 100)
    print ''



def entropy_by_compression_ratio_interpolator():
    """Interpolation of the entropy that leads to a specific comperssion ratio. """

    entropies = np.linspace(0, 1, 100)
    compression_ratios = np.array([ 0.00113678,  0.01353455,  0.02642822,  0.03891754,  0.05001831,
                                    0.06508636,  0.07836151,  0.09088135,  0.10540771,  0.11664581,
                                    0.12976837,  0.14237213,  0.15589905,  0.16797638,  0.17984772,
                                    0.19004059,  0.20398712,  0.2146225 ,  0.22814941,  0.23826599,
                                    0.25154877,  0.26065826,  0.2771225 ,  0.28622437,  0.29885864,
                                    0.30922699,  0.32141876,  0.33226776,  0.34382629,  0.35559082,
                                    0.36676025,  0.37932587,  0.38925171,  0.39923096,  0.41022491,
                                    0.42311096,  0.43325806,  0.44380188,  0.45388031,  0.46531677,
                                    0.47728729,  0.48655701,  0.49632263,  0.50746155,  0.51893616,
                                    0.53119659,  0.54163361,  0.55183411,  0.56119537,  0.57220459,
                                    0.58285522,  0.59011078,  0.60164642,  0.6098175 ,  0.62010956,
                                    0.62983704,  0.64057159,  0.65023804,  0.65835571,  0.66780853,
                                    0.67849731,  0.68518829,  0.69530487,  0.70523071,  0.71391296,
                                    0.72187805,  0.72968292,  0.73725891,  0.74497986,  0.75293732,
                                    0.76076508,  0.76959229,  0.77527618,  0.78244781,  0.78964233,
                                    0.79663849,  0.80402374,  0.81163025,  0.8180542 ,  0.82398224,
                                    0.83235168,  0.83975983,  0.84716034,  0.85497284,  0.86434937,
                                    0.87173462,  0.88202667,  0.89076233,  0.90008545,  0.90943909,
                                    0.92072296,  0.92938232,  0.93776703,  0.94753265,  0.9566803 ,
                                    0.96694183,  0.97696686,  0.98638916,  0.99610138,  1.00035095])
    return scipy.interpolate.interp1d(compression_ratios, entropies, kind='slinear')    


def main():
    np.random.seed()
    parser = argparse.ArgumentParser(description='BLOB Generator')
    parser.add_argument('filename', nargs=1, help='filename')
    parser.add_argument('-u', '--uniques', dest='U', action='store', type=int, help='Number of unique blocks', required=True)
    parser.add_argument('-d', '--duplication', dest='D', action='store', type=int,
                        help='The size of the pool for block duplication, and the multiplicty (default: 0,0)', nargs=2, default=(0,0))

    prob = parser.add_mutually_exclusive_group(required=True)
    prob.add_argument('-e', '--entropy', dest='E', action='store', type=float, help='The (binary) entropy of the generated data (0.0-1.0)')
    prob.add_argument('-c', '--compression', dest='C', action='store', type=float, help='Compression ratio (approximated, 0.0035-1.0)')

    parser.add_argument('-s', '--block-size', dest='S', action='store', type=int, default=4, help='Block size in Kilobytes (default: 4)')


    args = parser.parse_args()

    if args.E is None:
        entropy_by_compression_ratio = entropy_by_compression_ratio_interpolator()
        entropy = entropy_by_compression_ratio(args.C)
    else:
        entropy = args.E

    generate(args.filename[0], args.S, args.U, args.D[0], args.D[1], entropy)

if __name__ == "__main__":
    main()



