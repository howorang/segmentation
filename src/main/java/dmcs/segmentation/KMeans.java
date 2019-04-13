package dmcs.segmentation;

import org.apache.commons.collections4.MultiMapUtils;
import org.apache.commons.collections4.MultiValuedMap;
import org.apache.commons.lang3.RandomUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.distance.ChebyshevDistance;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;

public class KMeans {
    private final int k;
    private final Database database;
    private Collection<RealVector> centroids = new ArrayList<>();
    private DistanceMeasure distanceMeasure = new ChebyshevDistance();
    private MultiValuedMap<RealVector, RealVector> centroidToPointsMap = MultiMapUtils.newSetValuedHashMap();

    public MultiValuedMap<RealVector, RealVector> getCentroidToPointsMap() {
        return centroidToPointsMap;
    }

    public KMeans(int k, Database database) {
        this.k = k;
        this.database = database;
    }

    public void execute(int iterations) {
        initCentroids();
        for (int i = 0; i < iterations; i++) {
            centroidToPointsMap = MultiMapUtils.newSetValuedHashMap();
            for (RealVector row : database.data) {
                centroidToPointsMap.put(assignCentroid(row), row);
            }
            List<RealVector> newCentroids = new ArrayList<>();
            centroidToPointsMap.asMap().forEach((centroid, points) -> newCentroids.add(computeNewCentroid(points)));
            centroids = newCentroids;
        }
    }

    private RealVector computeNewCentroid(Collection<RealVector> points) {
        return points.stream()
                .reduce(RealVector::add)
                .map(sum -> sum.mapDivide(points.size()))
                .orElseThrow();
    }

    private RealVector assignCentroid(RealVector row) {
        return centroids.stream()
                .map(centroid -> Pair.of(centroid, distanceMeasure.compute(centroid.toArray(), row.toArray())))
                .sorted(Comparator.comparing(Pair::getRight))
                .limit(1)
                .findAny()
                .orElseThrow()
                .getLeft();
    }


    private void initCentroids() {
        for (int i = 0; i < k; i++) {
            centroids.add(database.data.get(RandomUtils.nextInt(0, database.data.size())));
        }
    }
}
