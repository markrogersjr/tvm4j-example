import os
import sys
import nnvm
from mxnet.gluon.model_zoo.vision import get_model

def main():
  path = sys.argv[1]
  name = sys.argv[2]
  block = get_model(name, pretrained=True)
  sym, params = nnvm.frontend.from_mxnet(block)
  sym = nnvm.sym.softmax(sym)
  target = 'llvm -mcpu=skylake-avx512 --system-lib'
  shape_dict = {'data': (1, 3, 224, 224)}
  graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
  dirname = '/home/ubuntu/tvm4j-example/bin/libs'
  if not os.path.isdir(dirname):
      os.makedirs(dirname)
  with open(os.path.join(dirname, 'net.json'), 'w') as f:
      f.write(graph.json())
  lib.export_library(os.path.join(dirname, 'net.so'))
  with open(os.path.join(dirname, 'net.params'), 'wb') as f:
      f.write(nnvm.compiler.save_param_dict(params))

if __name__ == '__main__':
  main()
