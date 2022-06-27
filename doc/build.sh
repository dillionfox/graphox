#/bin/bash

dir=$(cd ../../ && pwd)

case ":$dir:" in
  *:$PYTHONPATH:*) true;;
  *) export PYTHONPATH=${dir}:${PYTHONPATH}: ;;
esac

pdoc --html graph_curvature
