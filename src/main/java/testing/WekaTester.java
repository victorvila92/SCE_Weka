package testing;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;

public class WekaTester {

    private static final String FILEPATH_TRAIN_CLICKS = "/home/victor/Desktop/sce_train_20190210_clicks.arff";
    private static final String FILEPATH_TRAIN_CONVERSION_RATE = "/home/victor/Desktop/sce_train_20190210_convRate.arff";
    private static final String FILEPATH_TEST_CLICKS = "/home/victor/Desktop/sce_test_20190210_clicks.arff";
    private static final String FILEPATH_TEST_CONVERSION_RATE = "/home/victor/Desktop/sce_test_20190210_convRate.arff";

    public static void main(String[] args){
        System.out.println("Initializing");

        try {

            Instances dataTrainClicks = getInstancesFromFile(FILEPATH_TRAIN_CLICKS);
            Instances dataTrainConvRate = getInstancesFromFile(FILEPATH_TRAIN_CONVERSION_RATE);
            Instances dataTestClicks = getInstancesFromFile(FILEPATH_TEST_CLICKS);
            Instances dataTestConvRate = getInstancesFromFile(FILEPATH_TEST_CONVERSION_RATE);

            assert dataTrainClicks != null;
            assert dataTrainConvRate != null;
            assert dataTestClicks != null;
            assert dataTestConvRate != null;

            SimpleKMeans skmClicks = createSimpleKMeansCluster(dataTrainClicks);
            SimpleKMeans skmConvRate = createSimpleKMeansCluster(dataTrainConvRate);

            ClusterEvaluation clsClicks = new ClusterEvaluation();
            clsClicks.setClusterer(skmClicks);
            clsClicks.evaluateClusterer(dataTestClicks);

            System.out.println(clsClicks.clusterResultsToString());

            ClusterEvaluation clsConvRate = new ClusterEvaluation();
            clsConvRate.setClusterer(skmConvRate);
            clsConvRate.evaluateClusterer(dataTestConvRate);

            System.out.println(clsConvRate.clusterResultsToString());

            System.out.println("----------------------------");

            double[] clicksResults = clsClicks.getClusterAssignments();

            for(int i = 0; i < clicksResults.length; i++) {
                System.out.println("Instance " + i + "  with " + dataTestClicks.get(i) + " clicks --> " +
                        "assigned to cluster: " + clicksResults[i] + " with mean " + skmClicks.getClusterCentroids().get((int)clicksResults[i]) + " clicks.");
            }

            System.out.println("----------------------------");

            double[] convRateResults = clsConvRate.getClusterAssignments();

            for(int i = 0; i < convRateResults.length; i++) {
                System.out.println("Instance " + i + "  with " + dataTestConvRate.get(i) + " convRate --> " +
                        "assigned to cluster: " + convRateResults[i] + " with mean " + skmConvRate.getClusterCentroids().get((int) convRateResults[i]) + " convRate.");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }


    }

    private static SimpleKMeans createSimpleKMeansCluster(Instances dataTrain){
        SimpleKMeans skmClicks =new SimpleKMeans();

        try{
            skmClicks.setNumClusters(12);
            skmClicks.buildClusterer(dataTrain);
            skmClicks.getClusterCentroids().sort(0);
            System.out.println(skmClicks);
        }catch (Exception e){
            e.printStackTrace();
        }

        return skmClicks;
    }

    private static Instances getInstancesFromFile(String filepath){
        ArffLoader loader = new ArffLoader();
        try {
            loader.setFile(new File(filepath));
            return loader.getDataSet();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}
