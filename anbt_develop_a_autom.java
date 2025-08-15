import java.io.IOException;
import org.apache.commons.io.FilenameUtils;
import org.json.JSONObject;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class anbt_develop_a_autom {

    public static void main(String[] args) throws Exception {
        // Load dataset
        Instances data = DataSource.read("data.arff");
        data.setClassIndex(data.numAttributes() - 1);

        // Define machine learning model
        J48 tree = new J48();
        tree.buildClassifier(data);

        // Evaluate model
        Evaluation evaluation = new Evaluation(data);
        evaluation.evaluateModel(tree, data);

        // Extract model details
        String modelDetails = tree.toString();
        JSONObject jsonData = new JSONObject();
        jsonData.put("model", modelDetails);
        jsonData.put("accuracy", evaluation.accuracy());
        jsonData.put("precision", evaluation.precision(0));
        jsonData.put("recall", evaluation.recall(0));

        // Output model details
        System.out.println("Model Details: " + jsonData.toString());
    }
}