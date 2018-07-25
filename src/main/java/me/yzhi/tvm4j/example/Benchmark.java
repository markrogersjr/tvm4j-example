package me.yzhi.tvm4j.example;

import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;
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

public class Benchmark {

  static BufferedImage img;

  public static void main(String[] args) throws IOException {

    String loadingDir = args[0];
    Module libmod = Module.load(loadingDir + File.separator + "deploy_lib.so");
    String graphJson = new Scanner(new File(loadingDir + File.separator + "deploy_graph.json"))
        .useDelimiter("\\Z").next();
    byte[] params = readBytes(loadingDir + File.separator + "deploy_param.params");

    TVMContext ctx = TVMContext.cpu();

    GraphRuntime.GraphModule graph = GraphRuntime.create(graphJson, libmod, ctx);

    FileReader fileReader = new FileReader("/home/ubuntu/labels.txt");
    BufferedReader bufferedReader = new BufferedReader(fileReader);
    int[] labels = new int[1000];
    int i = 0;
    String line = null;
    while ((line = bufferedReader.readLine()) != null) {
        labels[i] = Integer.parseInt(line, 10); // newline char at end of line?
        i += 1;
    }
    float count = 0;
    float top1 = 0;
    float top5 = 0;
    int best = 0;
    float bestVal = Float.NEGATIVE_INFINITY;
    for (i=0; i<1000; i++) {
      String path = String.format("/home/ubuntu/imagenet1000/ILSVRC2012_val_%08d.JPEG", i + 1);
      graph.loadParams(params).setInput("data", ImageArray(path)).run();
      NDArray output = NDArray.empty(new long[]{1, 1000});
      float[] outputArr = output.asFloatArray();
      graph.getOutput(0, output);
      best = 0;
      System.out.println(String.format("labels[%d] = %d", i, labels[i]));
      for (int j=0; j<1000; j++) {
        System.out.println(String.format("out[%d] = %f", j, outputArr[j]));
        if (outputArr[i] > bestVal) {
          best = j;
          bestVal = outputArr[i];
        }
      }
      if (labels[i] == best) {
        top1 += 1;
      }
        // TODO: argsort to get top5?
    }
    top1 /= 1000;
    System.out.println(top1);
    
//    graph.loadParams(params).setInput("data", RandomInput()).run();
//
//    NDArray output = NDArray.empty(new long[]{1, 1000});
//    graph.getOutput(0, output);
//
//    float[] outputArr = output.asFloatArray();
//    for (int i = 0; i < 10; ++i) {
//      System.out.println(outputArr[i]);
//    }

    System.out.println("Done.");
  }

  public static byte[] readBytes(String filename) throws IOException {
    File file = null;
    FileInputStream fileStream = new FileInputStream(file = new File(filename));
    byte[] arr = new byte[(int) file.length()];
    fileStream.read(arr, 0, arr.length);
    return arr;
  }

  private static NDArray RandomInput() {
    float[] arr = new float[1*3*224*224];
    for (int i = 0; i < arr.length; ++i) {
      arr[i] = (float) Math.random();
    }
    NDArray nd = NDArray.empty(new long[]{1, 3, 224, 224});
    nd.copyFrom(arr);
    return nd;
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
    int rgb;
    float alpha;
    float red;
    float green;
    float blue;
    for (int i = 0; i < 224; i++) {
      for (int j = 0; j < 224; j++) {
        rgb = img.getRGB(i, j); //always returns TYPE_INT_ARGB
        alpha = (float) ((rgb >> 24) & 0xFF);
        red =   (float) ((rgb >> 16) & 0xFF);
        green = (float) ((rgb >>  8) & 0xFF);
        blue =  (float) ((rgb      ) & 0xFF);
        arr[3 * 224 * 0 + 224 * i + j] = (float) ((red - 123) / 58.395);
        arr[3 * 224 * 1 + 224 * i + j] = (float) ((green - 117) / 57.12);
        arr[3 * 224 * 2 + 224 * i + j] = (float) ((blue - 104) / 57.375);
      }
    }
    NDArray nd = NDArray.empty(new long[]{1, 3, 224, 224});
    nd.copyFrom(arr);
    return nd;
  }

}
