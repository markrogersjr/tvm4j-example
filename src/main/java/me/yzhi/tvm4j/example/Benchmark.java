package me.yzhi.tvm4j.example;

import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;
import ml.dmlc.tvm.contrib.GraphModule;
import ml.dmlc.tvm.contrib.GraphRuntime;

import java.io.*;
import java.util.Scanner;

import static me.yzhi.tvm4j.example.GraphForward.readBytes;

public class Benchmark {
  public static void main(String[] args) throws IOException {

    String loadingDir = args[0];
    Module libmod = Module.load(loadingDir + File.separator + "net.so");
    String graphJson = new Scanner(new File(loadingDir + File.separator + "net.json"))
        .useDelimiter("\\Z").next();
    byte[] params = readBytes(loadingDir + File.separator + "net.params");

    TVMContext ctx = TVMContext.cpu();

    GraphModule graph = GraphRuntime.create(graphJson, libmod, ctx);

//    graph.loadParams(params).setInput("data", FixedInput()).run();
    graph.loadParams(params).setInput("data", ArrayInput(1)).run();
    NDArray output = NDArray.empty(new long[]{1, 1000});
    graph.getOutput(0, output);
    float[] outputArr = output.asFloatArray();
    write("out_java.txt", outputArr);
    System.out.println("Done.");
  }

  private static NDArray FixedInput() throws IOException {
//    FileReader fileReader = new FileReader("arr.txt");
//    BufferedReader bufferedReader = new BufferedReader(fileReader);
//    float[] arr = new float[3 * 224 * 224];
//
//    int i = 0;
//    String line = null;
//    while ((line = bufferedReader.readLine()) != null) {
//      arr[i] = Float.parseFloat(line);
//      i += 1;
//    }
    float[] arr = read_floats("arr.txt", 3 * 224 * 224);
    NDArray nd = NDArray.empty(new long[]{1, 3, 224, 224});
    nd.copyFrom(arr);
    return nd;
  }

  private static NDArray ArrayInput(int index) throws IOException {
    float[] arr = read_floats(String.format("%d.txt", index), 3 * 224 * 224);
    NDArray nd = NDArray.empty(new long[]{1, 3, 224, 224});
    nd.copyFrom(arr);
    return nd;
  }

  public static float[] read_floats(String filename, int length) throws IOException{
    FileReader fileReader = new FileReader(filename);
    BufferedReader bufferedReader = new BufferedReader(fileReader);
    float[] arr = new float[length];
    String line = null;
    for (int i=0; i<length; i++) {
      line = bufferedReader.readLine();
      arr[i] = Float.parseFloat(line);
    }
    return arr;
  }

  public static int[] read_ints(String filename, int length) throws IOException{
    FileReader fileReader = new FileReader(filename);
    BufferedReader bufferedReader = new BufferedReader(fileReader);
    int[] arr = new int[length];
    String line = null;
    for (int i=0; i<length; i++) {
      line = bufferedReader.readLine();
      arr[i] = Integer.parseInt(line);
    }
    return arr;
  }

  public static void write(String filename, float[]x) throws IOException{
    BufferedWriter outputWriter = null;
    outputWriter = new BufferedWriter(new FileWriter(filename));
    for (int i = 0; i < x.length; i++) {
      outputWriter.write(String.format("%.12f\n", x[i]));
    }
    outputWriter.flush();
    outputWriter.close();
  }

  public static void write(String filename, int[]x) throws IOException{
    BufferedWriter outputWriter = null;
    outputWriter = new BufferedWriter(new FileWriter(filename));
    for (int i = 0; i < x.length; i++) {
      outputWriter.write(String.format("%d\n", x[i]));
    }
    outputWriter.flush();
    outputWriter.close();
  }

}
