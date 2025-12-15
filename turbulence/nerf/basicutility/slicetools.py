from typing import Sequence


def indices2slice(indices, axis):
    """
    converting slice style strings to a list of slices based on the position of the axis
    indices: list of indices
    axis: list of axis, axis' elements need to be in the ascending order and non-negative
    
    Example
    ------
    >>> indices2slice([[1,2],"3:-1:2"], [1,3])
    [slice(None), sliceslice(1, 2, None), slice(None), slice(3, -1, 2)]
    # the output is equivalent to: arr[:, [1,2], :, 3:-1:2], arr is a numpy array
    """
    
    assert all(ax0 < ax1 for ax0, ax1 in zip(axis[:-1], axis[1:])), "Elements in axis must be in ascending order"

    slices = []
    for axis_slice in indices:
        if isinstance(axis_slice, str):
            indice = slice(*[None if i=="" else int(i) for i in axis_slice.split(":")])
        else:
            indice = axis_slice # for [advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)
        slices.append(indice)
    
    slices = iter(slices)

    results = []
    for ax in range(axis[-1]+1):
        if ax in axis:
            results.append(next(slices))
        else:
            results.append(slice(None))

    return results

