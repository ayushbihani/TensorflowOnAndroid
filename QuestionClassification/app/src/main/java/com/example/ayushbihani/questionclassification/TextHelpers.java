package com.example.ayushbihani.questionclassification;

import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Created by ayushbihani on 3/4/18.
 */

public class TextHelpers {
    private HashMap<String, Integer> word2index;
    private Set<String> stopwords;
    private AssetManager assetManager;
    private Pattern pattern;
    private int inputSize;
    private final String STOPWORDS_FILE_PATH = "stopwords.txt";
    private final String VOCABULART_FILE_NAME = "vocabulary.txt";

    TextHelpers(AssetManager assetManager) {
        this.assetManager = assetManager;
        this.stopwords = new HashSet<>();
        init();
    }

    public void init() {
        word2index = new HashMap<>();
        stopwords = new HashSet<>();
        readStopWords(STOPWORDS_FILE_PATH);
        readVocabalury(VOCABULART_FILE_NAME);
    }

    /**
     * This function creates a regular expression patter for eliminating stop words
     * from a string text.
     *
     * @param file input file
     */
    public void readStopWords(String file) {
        BufferedReader bufferedReader = streamReader(file);
        String line = "";
        StringBuilder stringBuilder = new StringBuilder("\\b(");
        try {
            while ((line = bufferedReader.readLine()) != null) {
                stringBuilder.append(line + "|");
            }
            stringBuilder.delete(stringBuilder.length() - 1, stringBuilder.length());
            stringBuilder.append(")\\b\\s?");
        } catch (IOException e) {
            e.printStackTrace();
        }
        pattern = Pattern.compile(stringBuilder.toString());
    }

    /**
     * This function creates a mapping between word and index
     *
     * @param file Input file to read vocabulary from
     */
    public void readVocabalury(String file) {
        BufferedReader br = streamReader(file);
        InputStream input;
        String line = "";
        try {
            input = assetManager.open(file);
            InputStreamReader inputreader = new InputStreamReader(input);
            BufferedReader bufferedreader = new BufferedReader(inputreader);
            while ((line = bufferedreader.readLine()) != null) {
                String[] pair = line.split(":");
                word2index.put(pair[0], word2index.size());
            }
            inputSize = word2index.size();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Returns a stream to the input file
     *
     * @param file Input file
     */
    public BufferedReader streamReader(String file) {
        BufferedReader br = null;
        try {
            InputStream input = assetManager.open(file);
            InputStreamReader inputStreamReader = new InputStreamReader(input);
            br = new BufferedReader(inputStreamReader);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return br;
    }

    public int getInputSize(){
        return inputSize;
    }

    /**
     * This function cleans text and retains main context of the text
     *
     * @param text Input text to clean
     */
    public String[] cleanText(String text) {
        text = text.replaceAll("\\d", "");
        text = text.replaceAll("[^a-zA-Z]+", " ");
        text = text.toLowerCase();
        text = text.toLowerCase();
        String cleanedText = pattern.matcher(text).replaceAll("");
        String[] textTokens = cleanedText.split(" ");
        int textWordCount = textTokens.length;
        if (textWordCount > inputSize) {
            ArrayList<String> list = new ArrayList<String>(Arrays.asList(textTokens));
            String[] stringList = new String[inputSize];
            stringList = list.subList(0, inputSize).toArray(stringList);
            return stringList;
        }
        return textTokens;
    }

    /**
     * This function maps a text to its indexed(integer) representation
     *
     * @param text input text to port to a integer representation
     */

    public float[] getVector(String text) {
        String[] cleanedText = cleanText(text);
        float[] wordVector = new float[inputSize];
        for (int i = 0; i < cleanedText.length; i++) {
            if (word2index.containsKey(cleanedText[i])) {
                wordVector[word2index.get(cleanedText[i])] += 1;
            }
        }
        return wordVector;
    }
}

