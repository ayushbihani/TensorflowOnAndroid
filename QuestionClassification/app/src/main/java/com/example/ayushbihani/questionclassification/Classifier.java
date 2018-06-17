package com.example.ayushbihani.questionclassification;

/**
 * Created by ayushbihani on 19/4/18.
 */

public interface Classifier {
    public class Recognition{
        String title;
        Float confidence;

        public Recognition(String title, Float confidence){
            this.title = title;
            this.confidence = confidence;
        }
        public String getTitle() {
            return title;
        }

        public void setTitle(String title) {
            this.title = title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public void setConfidence(Float confidence) {
            this.confidence = confidence;
        }
    }
    Recognition recognizeText(String text);
}
