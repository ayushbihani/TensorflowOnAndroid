package com.example.ayushbihani.questionclassification;

import android.content.res.AssetManager;
import android.os.Trace;
import android.util.Log;

import com.example.ayushbihani.questionclassification.Classifier;
import com.example.ayushbihani.questionclassification.TextHelpers;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Vector;

/**
 * Created by ayushbihani on 3/4/18.
 */

public class TensorFlowClassifier implements Classifier {

    private static final String TAG = "TensorFlowClassifier";
    private String inputName;
    private String[] outputName;
    private int inputSize;
    private static TextHelpers textHelper;
    private float[] outputValues;
    private static Vector<String> labels = new Vector<>();
    private TensorFlowInferenceInterface inferenceInterface;
    private TensorFlowClassifier(){}

    /**
     * @param assetManager  The asset manager to be used to load assets.
     * @param inputName The label of the input node.
     * @param modelFileName The filepath of the model GraphDef protocol buffer.
     * @param outputName The label of the output node.
     * */

    public static Classifier create(
            AssetManager assetManager,
            String modelFileName,
            String inputName,
            String[] outputName){

        TensorFlowClassifier classifier = new TensorFlowClassifier();
        classifier.inputName = inputName;
        classifier.outputName = outputName;
        addLabels();
        classifier.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFileName);
        textHelper = new TextHelpers(assetManager);
        classifier.inputSize = textHelper.getInputSize();
        final int numClasses = 6;
        classifier.outputValues = new float[numClasses];
        return classifier;
    }

    /**
     * Ideally we should read labels from a text file. Since we have
     * few labels, we are adding it individually.
     * label_map = {"DESC":0, "ENTY":1, "HUM":2, "NUM":3, "LOC":4, "ABBR":5}
     * */
    public static void addLabels(){
        labels.add(0,"DESC");
        labels.add(1,"ENTY");
        labels.add(2,"HUM");
        labels.add(3,"NUM");
        labels.add(4,"LOC");
        labels.add(5,"ABBR");
    }

    @Override
    public Recognition recognizeText(String text) {
        Trace.beginSection("Recognize");
        Trace.beginSection("Process text");

        float[] vector = textHelper.getVector(text);
        Trace.endSection();

        Trace.beginSection("Feeding data");
        inferenceInterface.feed(inputName, vector, 1, inputSize);
        Trace.endSection();

        Trace.beginSection("running inference");
        inferenceInterface.run(outputName);
        Trace.endSection();

        Trace.beginSection("Fetch");
        inferenceInterface.fetch(outputName[0], outputValues);

        float probability = -1;
        int index = -1;
        for(int i = 0; i < outputValues.length; i++){
            float temp = outputValues[i];
            if(temp > probability){
                probability = temp;
                index = i;
            }
        }
        Trace.endSection();
        return new Recognition(labels.get(index),outputValues[index]);
    }


    public void close(){
        inferenceInterface.close();
    }
}
