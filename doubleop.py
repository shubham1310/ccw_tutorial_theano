from theano import Op, Apply
from theano.tensor import as_tensor_variable

class DoubleOp(Op):
    __props__ = ()

    def make_node(self, x):
        x = as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = x * 2

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def grad(self, inputs, output_grads):
        return [output_grads[0] * 2]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


class DoubleC:
    # This C code only works for 1D
    def c_code(self, node, name, input_names,
               output_names, sub):
        return """
Py_XDECREF(%(out)s);
%(out)s = (PyArrayObject *)PyArray_NewLikeArray(
    %(inp)s, NPY_ANYORDER, NULL, 0);
if (%(out)s == NULL) {
  %(fail)s
}
{
PyObject *iter_in = PyArray_IterNew(inp),
PyObject *iter_out = PyArray_IterNew(*out);
if (iter_in == NULL || iter_out == NULL) {
  Py_XDECREF(iter_in); Py_XDECREF(iter_out);
  %(fail)s
}
for (;PyArray_ITER_NOTDONE(iter_in) &&
       PyArray_ITER_NOTDONE(iter_out);
     PyArray_ITER_NEXT(iter_in),
      PyArray_ITER_NEXT(iter_out)) {
  *(dtype_%(out)s *)PyArray_ITER_DATA(iter_out) =
    (*(dtype_%(inp)s *)PyArray_ITER_DATA(iter_in)) * 2;
}
Py_DECREF(iter_in);
Py_DECREF(iter_out);
}
""" % dict(inp=input_names[0], out=output_names[0],
           fail=sub["fail"])


from theano.gof import COp

class DoubleCOp(COp):
    __props__ = ()

    def __init__(self):
        COp.__init__(self, "./doublecop.c",
                     "APPLY_SPECIFIC(doublecop)")

    def make_node(self, x):
        x = as_tensor_variable(x)
        return Apply(self, [x], [x.type()])
