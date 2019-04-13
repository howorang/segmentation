package dmcs.segmentation;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Database {

    public final List<RealVector> data;

    public Database(String filePath) throws IOException {
        data = Files.lines(Path.of(filePath))
                .map(line -> line.substring(1, line.length() - 1))
                .map(line -> StringUtils.split(line, ","))
                .map(this::convertToDouble)
                .map(ArrayRealVector::new)
                .collect(Collectors.toUnmodifiableList());

    }

    private double[] convertToDouble(String[] row) {
        return Stream.of(row).map(Double::parseDouble).mapToDouble(p -> p).toArray();
    }
}
