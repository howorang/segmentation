package dmcs.segmentation;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.DefaultXYDataset;

import javax.swing.*;
import java.util.List;

public class LineChart extends JFrame {
    public LineChart(String applicationTitle, String chartTitle, List<History> history) {
        super(applicationTitle);
        // This will create the dataset
        DefaultXYDataset dataset = createDataset(history);
        // based on the dataset we create the chart
        JFreeChart chart = createChart(dataset, chartTitle);
        // we put the chart into a panel
        ChartPanel chartPanel = new ChartPanel(chart);
        // default size
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        // add it to our application
        setContentPane(chartPanel);

    }

    /**
     * Creates a sample dataset
     */
    private DefaultXYDataset createDataset(List<History> history) {
        DefaultXYDataset result = new DefaultXYDataset();
        double[][] centroidSumSeries = extractCentroidSum(history);
        double[][] distSumSeries = extractDistSumSeries(history);
        result.addSeries("CentroidSum", centroidSumSeries);
        result.addSeries("DistSum", distSumSeries);
        return result;

    }

    private double[][] extractDistSumSeries(List<History> history) {
        double[][] res = new double[2][];
        res[0] = new double[history.size()];
        res[1] = new double[history.size()];
        for (int i = 0; i < history.size(); i++) {
            res[0][i] = (double) i;
            res[1][i] = history.get(i).getDistanceSum();
        }
        return res;
    }

    private double[][] extractCentroidSum(List<History> history) {
        double[][] res = new double[2][];
        res[0] = new double[history.size()];
        res[1] = new double[history.size()];
        for (int i = 0; i < history.size(); i++) {
            res[0][i] = (double) i;
            res[1][i] = history.get(i).getCentroidSum();
        }
        return res;
    }

    /**
     * Creates a chart
     */
    private JFreeChart createChart(DefaultXYDataset dataset, String title) {

        JFreeChart chart = ChartFactory.createXYLineChart(
                title,                  // chart title
                "",
                "",
                dataset                // data

        );

        XYPlot plot = (XYPlot) chart.getPlot();
        plot.setForegroundAlpha(0.5f);
        return chart;

    }

}
