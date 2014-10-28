from scalmulop import ScalMulV1
from doubleop import DoubleOp
from theano.compile import optdb

from theano.gof import local_optimizer

@local_optimizer([ScalMulV1])
def local_scalmul_double_v1(node):
    if not (isinstance(node.op, ScalMulV1)
            and node.op.scal == 2):
        return False

    return DoubleOp()(node.inputs[0])

from theano.gof.opt import OpSub

local_scalmul_double_v2 = OpSub(ScalMulV1(2), DoubleOp())

optdb['specialize'].register(
    # name of optimization (must be unique)
    'local_scalmul_double_v2',
    # optimization function
    local_scalmul_double_v2,
    # tags to activate/deactivate as a group
    'fast_run')
