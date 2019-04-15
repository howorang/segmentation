package dmcs.segmentation;

import org.apache.commons.collections4.MultiMapUtils;
import org.apache.commons.collections4.MultiValuedMap;
import org.apache.commons.lang3.RandomUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.distance.ChebyshevDistance;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;

public class KMeans {
    private final int k;
    private final Database database;
    private Collection<RealVector> centroids = new ArrayList<>();
    private DistanceMeasure distanceMeasure = new EuclideanDistance();
    private MultiValuedMap<RealVector, RealVector> centroidToPointsMap = MultiMapUtils.newSetValuedHashMap();
    private boolean collectHistory;
    private History history;
    private List<Double> distances;

    public MultiValuedMap<RealVector, RealVector> getCentroidToPointsMap() {
        return centroidToPointsMap;
    }

    public KMeans(int k, Database database) {
        this.k = k;
        this.database = database;
    }

    public void setCollectHistory(boolean collectHistory) {
        this.collectHistory = collectHistory;
    }

    public History getHistory() {
        return history;
    }

    public void execute(int iterations) {
        if (collectHistory) {
            history = new History();
            distances = new ArrayList<>();
        }
        initCentroids();
        for (int i = 0; i < iterations; i++) {
            centroidToPointsMap = MultiMapUtils.newSetValuedHashMap();
            for (RealVector row : database.data) {
                centroidToPointsMap.put(assignCentroid(row), row);
            }
            computeNewCentroids();
        }
        if (collectHistory) {
            history.setDistanceSum(computeDistanceSum());
            history.setCentroidSum(computeCentroidSum());
        }
    }

    private double computeDistanceSum() {
        double distanceSum = 0;
        for (var centroidMapEntry : centroidToPointsMap.asMap().entrySet()) {
            for (var point : centroidMapEntry.getValue()) {
                distanceSum += distanceMeasure.compute(point.toArray(), centroidMapEntry.getKey().toArray());
            }
        }
        return distanceSum;
    }

    private double computeCentroidSum() {
        double centroidSum = 0;
        if (centroids.size() == 1) {
            return 0;
        }
        for (RealVector cenI : centroids) {
            centroidSum += centroids.stream()
                    .filter(cenJ -> !cenJ.equals(cenI))
                    .map(cenJ -> distanceMeasure.compute(cenJ.toArray(), cenI.toArray()))
                    .mapToDouble(p -> p)
                    .min()
                    .orElseThrow();
        }
        return centroidSum;
    }

    private void computeNewCentroids() {
        List<RealVector> newCentroids = new ArrayList<>();
        centroidToPointsMap.asMap().forEach((centroid, points) -> newCentroids.add(computeNewCentroid(points)));
        centroids = newCentroids;
    }

    private RealVector computeNewCentroid(Collection<RealVector> points) {
        return points.stream()
                .reduce(RealVector::add)
                .map(sum -> sum.mapDivide(points.size()))
                .orElseThrow();
    }

    private RealVector assignCentroid(RealVector row) {
        return centroids.stream()
                .map(centroid -> Pair.of(centroid, getDistance(row, centroid)))
                .sorted(Comparator.comparing(Pair::getRight))
                .limit(1)
                .findAny()
                .orElseThrow()
                .getLeft();
    }

    private double getDistance(RealVector row, RealVector centroid) {
        return distanceMeasure.compute(centroid.toArray(), row.toArray());
    }


    private void initCentroids() {
        for (int i = 0; i < k; i++) {
            centroids.add(database.data.get(RandomUtils.nextInt(0, database.data.size())));
        }
    }
}
