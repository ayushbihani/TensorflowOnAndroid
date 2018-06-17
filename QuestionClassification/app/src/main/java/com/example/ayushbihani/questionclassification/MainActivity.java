package com.example.ayushbihani.questionclassification;

import android.app.Activity;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    private final String TAG = "MainActivity";
    private final String INPUT_NAME = "inputTensor";
    private Classifier classifier;
    private EditText editText;
    private Button button;
    private final String[] OUTPUT_NAME = {"output"};
    private final String MODEL_FILE = "file:///android_asset/optimizedtensorflowModel.pb";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        editText = (EditText) findViewById(R.id.text);
        button = (Button) findViewById(R.id.button);
        loadModel();
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    public void loadModel() {
        Thread loadmodel = new Thread(new Runnable() {
            @Override
            public void run() {
                classifier = TensorFlowClassifier.create(
                        MainActivity.this.getAssets(),
                        MODEL_FILE,
                        INPUT_NAME,
                        OUTPUT_NAME);
                if (classifier == null) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, "Failed To load Model", Toast.LENGTH_SHORT).show();
                        }
                    });

                } else {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, "Model Successfully loaded", Toast.LENGTH_SHORT).show();
                            performInference();
                        }
                    });
                }
            }
        });
        loadmodel.start();
    }

    public void performInference() {
        button.setEnabled(true);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String text = editText.getText().toString();
                if(classifier!=null){
                    Classifier.Recognition recognition = classifier.recognizeText(text);
                    Toast.makeText(MainActivity.this, "Predicted: " + recognition.getTitle() ,Toast.LENGTH_LONG).show();
                }
            }
        });
    }
}