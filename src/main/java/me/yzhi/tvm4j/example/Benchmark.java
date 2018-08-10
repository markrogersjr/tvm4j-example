package me.yzhi.tvm4j.example;

import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;
import ml.dmlc.tvm.contrib.GraphModule;
import ml.dmlc.tvm.contrib.GraphRuntime;

import java.io.*;
import java.util.Scanner;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import java.io.*;
import javax.imageio.*;
import javax.swing.*;

import java.time.Instant;
import java.time.Duration;

import static me.yzhi.tvm4j.example.GraphForward.readBytes;

public class Benchmark {

  static BufferedImage img;

  public static void main(String[] args) throws IOException {

    String loadingDir = args[0];
    Module libmod = Module.load(loadingDir + File.separator + "net.so");
    String graphJson = new Scanner(new File(loadingDir + File.separator + "net.json"))
        .useDelimiter("\\Z").next();
    byte[] params = readBytes(loadingDir + File.separator + "net.params");

    TVMContext ctx = TVMContext.cpu();

    GraphModule graph = GraphRuntime.create(graphJson, libmod, ctx);
    graph.loadParams(params);

    int[] labels = read_ints("/home/ubuntu/labels.txt", 1000);
    float accuracy = 0;
    float latency = 0;
    for (int i=0; i<1000; i++) {

      String path = String.format("/home/ubuntu/imagenet1000/ILSVRC2012_val_%08d.JPEG", i + 1);
      graph.setInput("data", ImageArray(path));

      Instant first = Instant.now();
      graph.run();
      Instant second = Instant.now();
      Duration duration = Duration.between(first, second);
      latency += (float) duration.toNanos() / 1000000;

      NDArray output = NDArray.empty(new long[]{1, 1000});
      graph.getOutput(0, output);
      float[] outputArr = output.asFloatArray();
      if (argmax(outputArr) == labels[i]) {
        accuracy += 1;
      }

    }
    accuracy *= 100.0 / 1000.0;
    write_floats("/home/ubuntu/accuracy.txt", new float[]{accuracy});
    latency /= 1000.0;
    write_floats("/home/ubuntu/latency.txt", new float[]{latency});

    System.out.println("Done.");
  }

  public static int argmax(float[] arr) {
    float bestval = Float.NEGATIVE_INFINITY;
    int index = 0;
    for (int i=0; i<arr.length; i++) {
      if (arr[i] > bestval) {
        index = i;
        bestval = arr[i];
      }
    }
    return index;
  }

  private static NDArray FixedInput() throws IOException {
    float[] arr = read_floats("arr.txt", 3 * 224 * 224);
    NDArray nd = NDArray.empty(new long[]{1, 3, 224, 224});
    nd.copyFrom(arr);
    return nd;
  }

  private static NDArray ArrayInput(int index) throws IOException {
    float[] arr = read_floats(String.format("/home/ubuntu/in/%d.txt", index), 3 * 224 * 224);
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

  public static void write_floats(String filename, float[]x) throws IOException{
    BufferedWriter outputWriter = null;
    outputWriter = new BufferedWriter(new FileWriter(filename));
    for (int i = 0; i < x.length; i++) {
      outputWriter.write(String.format("%.12f\n", x[i]));
    }
    outputWriter.flush();
    outputWriter.close();
  }

  public static void write_ints(String filename, int[]x) throws IOException{
    BufferedWriter outputWriter = null;
    outputWriter = new BufferedWriter(new FileWriter(filename));
    for (int i = 0; i < x.length; i++) {
      outputWriter.write(String.format("%d\n", x[i]));
    }
    outputWriter.flush();
    outputWriter.close();
  }

  public static BufferedImage toBufferedImage(Image img) {
    if (img instanceof BufferedImage) {
      return (BufferedImage) img;
    }
  
    // Create a buffered image with transparency
    BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);
  
    // Draw the image on to the buffered image
    Graphics2D bGr = bimage.createGraphics();
    bGr.drawImage(img, 0, 0, null);
    bGr.dispose();
  
    // Return the buffered image
    return bimage;
  }  

  private static NDArray ImageArray(String path) {
    try {
      img = toBufferedImage(ImageIO.read(new File(path)).getScaledInstance(224, 224, Image.SCALE_DEFAULT));
    } catch (IOException e) {
    }
    float[] arr = new float[3 * 224 * 224];
    float[] brr = new float[3 * 224 * 224];
    int rgb;
    int count = 0;
    float alpha;
    float red;
    float green;
    float blue;
    float[] colors;
    for (int i = 0; i < 224; i++) {
      for (int j = 0; j < 224; j++) {
        rgb = img.getRGB(i, j); //always returns TYPE_INT_ARGB
        alpha = (float) ((rgb >> 24) & 0xFF);
        red =   (float) ((rgb >> 16) & 0xFF);
        green = (float) ((rgb >>  8) & 0xFF);
        blue =  (float) ((rgb      ) & 0xFF);
        colors = new float[]{(float) ((red - 123) / 58.395), (float) ((green - 117) / 57.12), (float) ((blue - 104) / 57.375)};
        for (int k=0; k<3; k++) {
          arr[224 * 224 * k + 224 * j + i] = colors[k];
        }
      }
    }
    NDArray nd = NDArray.empty(new long[]{1, 3, 224, 224});
    nd.copyFrom(arr);
    return nd;
  }

}
