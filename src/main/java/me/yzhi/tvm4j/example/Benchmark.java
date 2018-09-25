package me.yzhi.tvm4j.example;

import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;
import ml.dmlc.tvm.contrib.GraphModule;
import ml.dmlc.tvm.contrib.GraphRuntime;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Scanner;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import java.io.*;
import javax.imageio.*;
import javax.swing.*;

import java.io.FileReader;
import java.io.BufferedReader;
import java.time.Instant;
import java.time.Duration;


public class Benchmark {

  static BufferedImage img;  

  public static void main(String[] args) throws IOException {

    String modelDirname = args[0];
    String imagesDirname = args[1];
    String tempDirname = args[2];
    String s3Dirname = args[3];
    int dim = Integer.parseInt(args[4]);
    Module libmod = Module.load(modelDirname + File.separator + "model.so");
    String graphJson = new Scanner(new File(modelDirname + File.separator + "model.json"))
        .useDelimiter("\\Z").next();
    byte[] params = readBytes(modelDirname + File.separator + "model.params");

    TVMContext ctx = TVMContext.cpu();

    int[] labels = read(tempDirname + File.separator + "labels.txt", 1000);
    float predictivity = 0;
    float latency = 0;
    float dt;

    GraphModule graph = GraphRuntime.create(graphJson, libmod, ctx);
    graph.loadParams(params);

    for (int i=0; i<1000; i++) {
      String image_filename = String.format("ILSVRC2012_val_%08d.JPEG", i + 1);
      String image_path = imagesDirname + File.separator + image_filename;
      NDArray x = ImageArray(image_path, dim);
      Instant startTime = Instant.now();
      graph.setInput("data", x);
      graph.run();
      NDArray y_arr = NDArray.empty(new long[]{1, 1000});
      graph.getOutput(0, y_arr);
      float[] y = y_arr.asFloatArray();
      Instant endTime = Instant.now();
      if (argMax(y) == labels[i])
        predictivity += 1;
      Duration duration = Duration.between(startTime, endTime);
      dt = (float) duration.toNanos() / 1000000;
      latency += dt;
      System.out.println(String.format("Predictivity = %.2f%% | Latency = %.4f ms (%d/%d)", 100 * predictivity / (i + 1), latency / (i + 1), i + 1, 1000));
    }
    predictivity *= 100 / 1000;
    latency /= 1000;
    write(tempDirname + File.separator + "predictivity.txt", new float[] {predictivity});
    write(tempDirname + File.separator + "latency.txt", new float[] {latency});
  }

  public static byte[] readBytes(String filename) throws IOException {
    File file = null;
    FileInputStream fileStream = new FileInputStream(file = new File(filename));
    byte[] arr = new byte[(int) file.length()];
    fileStream.read(arr, 0, arr.length);
    return arr;
  }

  public static int[] read(String path, int size) throws IOException{
    File file = new File(path);
    BufferedReader br = new BufferedReader(new FileReader(file));
    int[] arr = new int[size];
    String st;
    for (int i=0; i<size; i++)
      arr[i] = Integer.parseInt(br.readLine());
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

  public static BufferedImage toBufferedImage(Image img) {
    if (img instanceof BufferedImage)
      return (BufferedImage) img;
  
    // Create a buffered image with transparency
    BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);
  
    // Draw the image on to the buffered image
    Graphics2D bGr = bimage.createGraphics();
    bGr.drawImage(img, 0, 0, null);
    bGr.dispose();
  
    // Return the buffered image
    return bimage;
  }

  private static NDArray ImageArray(String path, int dim) {
    try {
      img = toBufferedImage(ImageIO.read(new File(path)).getScaledInstance(dim, dim, Image.SCALE_DEFAULT));
    } catch (IOException e) {
    }
    float[] arr = new float[3 * dim * dim];
    int rgb;
    int count = 0;
    float alpha;
    float red;
    float green;
    float blue;
    float[] colors;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        rgb = img.getRGB(i, j); //always returns TYPE_INT_ARGB
        alpha = (float) ((rgb >> 24) & 0xFF);
        red =   (float) ((rgb >> 16) & 0xFF);
        green = (float) ((rgb >>  8) & 0xFF);
        blue =  (float) ((rgb      ) & 0xFF);
        colors = new float[]{(float) ((red - 123) / 58.395), (float) ((green - 117) / 57.12), (float) ((blue - 104) / 57.375)};
        for (int k=0; k<3; k++)
          arr[dim * dim * k + dim * j + i] = colors[k];
      }
    }
    NDArray nd = NDArray.empty(new long[]{1, 3, dim, dim});
    nd.copyFrom(arr);
    return nd;
  }

  private static int argMax(float[] x) {
    int maxind = -1;
    float maxval = Float.NEGATIVE_INFINITY;
    for (int i=0; i< x.length; i++) {
      if (x[i] > maxval) {
        maxind = i;
        maxval = x[i];
      }
    }
    return maxind;
  }

}

