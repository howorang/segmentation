package dmcs.segmentation;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("Two arguments required: k and number of iterations");
        }
        Database database = new Database("C:\\Users\\Piotr Borczyk\\IdeaProjects\\segmentation\\src\\main\\resources\\segmentacja.txt");
        KMeans kMeans = new KMeans(Integer.parseInt(args[0]), database);
        kMeans.execute(Integer.parseInt(args[1]));
        kMeans.getCentroidToPointsMap().asMap().forEach((centroid, points) -> {
            System.out.println("CENTROID: " + centroid.toString());
            points.forEach(point -> System.out.println("    " + point));
        });
    }
}
