#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=$(cd $(dirname $0)/../; pwd)
mkdir -p $CURR_DIR/libs

echo "Run python compiler and dump library ..."
python benchmark.py $CURR_DIR/libs $1
echo $1

echo "Run java code ..."
CLASSPATH=$CLASSPATH:$PROJ_DIR/target/*:$PROJ_DIR/target/classes/lib/*
java -cp $CLASSPATH \
  -Dlog4j.configuration=file://$PROJ_DIR/conf/log4j.properties \
  me.yzhi.tvm4j.example.Benchmark $CURR_DIR/libs
