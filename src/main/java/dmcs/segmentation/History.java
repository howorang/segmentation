package dmcs.segmentation;

public class History {
    private double avgDistance;
    private double distanceSum;
    private double centroidSum;

    public Double getAvgDistance() {
        return avgDistance;
    }

    public void setAvgDistance(Double avgDistance) {
        this.avgDistance = avgDistance;
    }

    public Double getDistanceSum() {
        return distanceSum;
    }

    public void setDistanceSum(Double distanceSum) {
        this.distanceSum = distanceSum;
    }

    public double getCentroidSum() {
        return centroidSum;
    }

    public void setCentroidSum(double centroidSum) {
        this.centroidSum = centroidSum;
    }
}
