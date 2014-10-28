THEANO_APPLY_CODE_SECTION

int APPLY_SPECIFIC(doublecop)(PyArrayObject *x,
                              PyArrayObject **out) {
  Py_XDECREF(*out);
  *out = (PyArrayObject *)PyArray_NewLikeArray(
                           inp, NPY_ANYORDER, NULL, 0);
  if (*out == NULL)
    return -1;

  for (PyObject *iter_in = PyArray_IterNew(inp),
         PyObject *iter_out = PyArray_IterNew(*out);
       PyArray_ITER_NOTDONE(iter_in) &&
         PyArray_ITER_NOTDONE(iter_out);
       PyArray_ITER_NEXT(iter_in),
         PyArray_ITER_NEXT(iter_out)) {
    *(DTYPE_OUTPUT_0 *)PyArray_ITER_DATA(iter_out) = 
      (*(DTYPE_INPUT_0 *)PyArray_ITER_DATA(iter_in)) * 2;
  }
}
