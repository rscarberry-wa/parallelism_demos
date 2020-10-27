package com.rscarberry.parallelismdemos;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import reactor.util.function.Tuple2;
import reactor.util.function.Tuples;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DistanceComputationsDemo {

    public static void main(String[] args) {

        int numVectors = 100000;
        int numCenters = 20;
        int vectorLength = 100;
        int warmupRuns = 10;
        int timeingRuns = 50;
        int outerLoops = 10;

        // Create some test data. This is simulating computing the distances between
        // vectors in N-space and cluster centers. This is a task done in clustering methods,
        // such as k-means clustering.
        List<double[]> vectors = generateRandomVectors(numVectors, vectorLength, 1234L);
        List<double[]> centers = generateRandomVectors(numCenters, vectorLength, 4567L);

        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        List<Tuple2<String, Callable<List<double[]>>>> callableTuples = new ArrayList<>(List.of(
                // Have the single-threaded implementation first, because the times of the others
                // are compared against it.
                Tuples.of("single-threaded", () -> {
                    return computeDistancesSingleThreaded(vectors, centers);
                }),
                Tuples.of("reactor, per vector", () -> {
                    return computeDistancesInParallelUsingReactor(vectors, centers);
                }),
                Tuples.of("reactor, batch", () -> {
                    return computeDistancesInParallelUsingReactor2(vectors, centers);
                }),
                Tuples.of("parallel streams", () -> {
                    return computeDistancesInParallelUsingParallelStreams(vectors, centers);
                }),
                Tuples.of("completable futures", () -> {
                    return computesDistancesInParallelUsingCompletableFutures(vectors, centers, executorService);
                }),
                Tuples.of("callables", () -> {
                    return computeDistancesUsingCallables(vectors, centers, executorService);
                })
        ));

        final String singleThreadedName = callableTuples.get(0).getT1();

        // To ensure that all method produce the same results.
        Map<String, List<double[]>> nameToDistances = new HashMap<>();
        // If distances are unequal after the first loop, this will be set to false.
        boolean allEqual = true;

        try {

            Map<String, List<Double>> timesMap = new HashMap<>();

            for (int loop = 0; loop < outerLoops; loop++) {

                // Shuffle, because timing results are influenced by order.
                Collections.shuffle(callableTuples);

                for (Tuple2<String, Callable<List<double[]>>> tuple : callableTuples) {

                    Tuple2<List<double[]>, SummaryStatistics> statsTuple = timeTask(
                            tuple.getT2(), warmupRuns, timeingRuns
                    );

                    String name = tuple.getT1();

                    // In the first loop, stash the distances in order to check that each
                    // callable produces the same results.
                    if (loop == 0) {
                        nameToDistances.put(name, statsTuple.getT1());
                    }

                    double meanMsec = statsTuple.getT2().getMean();
                    timesMap.compute(name, (nm, lst) -> {
                        if (lst == null) {
                            lst = new ArrayList<>();
                        }
                        lst.add(meanMsec);
                        return lst;
                    });

                    SummaryStatistics summaryStatistics = statsTuple.getT2();
                    System.out.printf("%03d: %-30s: min = %d msec, max = %d msec, mean = %4.1f msec%n",
                            loop,
                            name,
                            (long) summaryStatistics.getMin(),
                            (long) summaryStatistics.getMax(),
                            summaryStatistics.getMean());

                }

                System.out.println();

                if (loop == 0) {
                    allEqual = checkDistancesEqual(singleThreadedName, nameToDistances);
                    // Not needed after first loop.
                    nameToDistances.clear();
                    // If the distances aren't all the same, bail out of the timing loops.
                    if (!allEqual) {
                        break;
                    }
                }
            }

            // Only show the final timing results if the distances were all equal.
            if (allEqual) {

                SortedMap<Double, String> avgTimes = new TreeMap<>();

                for (Map.Entry<String, List<Double>> entry : timesMap.entrySet()) {
                    String name = entry.getKey();
                    List<Double> times = entry.getValue();
                    double sum = 0.0;
                    for (Double time : times) {
                        sum += time;
                    }
                    avgTimes.put(sum / times.size(), name);
                }

                double singleThreadedMs = 0.0;
                for (Map.Entry<Double, String> entry : avgTimes.entrySet()) {
                    if (entry.getValue().equals("single-threaded")) {
                        singleThreadedMs = entry.getKey();
                        break;
                    }
                }

                System.out.printf("%nFinal Results:%n");

                final double stm = singleThreadedMs;

                avgTimes.entrySet()
                        .forEach(e -> {
                            double meanMs = e.getKey();
                            String name = e.getValue();
                            double speedUp = stm / meanMs;
                            System.out.printf("\t%s: %4.1f average msec, %3.1f speedup%n",
                                    name, meanMs, speedUp);
                        });
            }

        } catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            executorService.shutdownNow();
        }
    }

    /**
     * Times the execution of a task by running it sequentially a number of times.
     *
     * @param task a {@link Callable} to execute and time. This should be a stateless task that can
     *             be run many times giving the same result each time.
     * @param warmupRuns how many times to execute the task before collecting timing information.
     * @param timingRuns hom many executions of the task to time.
     * @param <T> what the task returns.
     * @return a {@link Tuple2} whose first element is the result from the task and whose second element
     *  is a {@link SummaryStatistics} instance containing the timing information in milliseconds.
     *  Its getMin() method returns the minimum number of milliseconds taken to execute the task. Its getMax()
     *  return the maximum number of milliseconds taken to execute the task. Finally, its getMean() method
     *  returns the mean number of milliseconds for task execution.
     * @throws Exception
     */
    public static <T> Tuple2<T, SummaryStatistics> timeTask(Callable<T> task, int warmupRuns, int timingRuns)
            throws Exception {
        for (int i = 0; i < warmupRuns; i++) {
            task.call();
        }
        SummaryStatistics summaryStatistics = new SummaryStatistics();
        T results = null;
        for (int j = 0; j < timingRuns; j++) {
            long startMs = System.currentTimeMillis();
            results = task.call();
            summaryStatistics.addValue(System.currentTimeMillis() - startMs);
        }
        return Tuples.of(results, summaryStatistics);
    }

    public static List<double[]> computeDistancesSingleThreaded(List<double[]> vectors, List<double[]> centers) {
        List<double[]> distances = new ArrayList<>(vectors.size());
        for (double[] vec : vectors) {
            double[] d = new double[centers.size()];
            for (int i = 0; i < d.length; i++) {
                d[i] = euclideanDistance(vec, centers.get(i));
            }
            distances.add(d);
        }
        return distances;
    }

    public static List<double[]> computeDistancesInParallelUsingReactor(
            List<double[]> vectors,
            List<double[]> centers) {

        final int numCenters = centers.size();

        List<Tuple2<Integer, double[]>> tuples = Flux.zip(
                Flux.range(0, vectors.size()), Flux.fromIterable(vectors))
                .flatMap(tuple -> Mono.just(tuple)
                        .subscribeOn(Schedulers.parallel())
                        .map(tpl -> {
                            double[] d = new double[numCenters];
                            for (int i = 0; i < numCenters; i++) {
                                d[i] = euclideanDistance(tpl.getT2(), centers.get(i));
                            }
                            return Tuples.of(tpl.getT1(), d);
                        }))
                .subscribeOn(Schedulers.immediate())
                .collectList()
                .block();

        tuples.sort((tpl1, tpl2) -> tpl1.getT1().compareTo(tpl2.getT1()));

        return tuples.stream().map(Tuple2::getT2).collect(Collectors.toList());
    }

    public static List<double[]> computeDistancesInParallelUsingReactor2(
            List<double[]> vectors,
            List<double[]> centers) {

        final int numVectors = vectors.size();
        final int numCenters = centers.size();

        final int[] vectorsPerThread = vectorsPerThread(numVectors);

        final int numProcessors = vectorsPerThread.length;

        List<Supplier<List<double[]>>> distanceSuppliers = new ArrayList<>(numProcessors);
        int startVector = 0;

        for (int i = 0; i < numProcessors; i++) {

            final int sv = startVector;
            final int nv = vectorsPerThread[i];

            distanceSuppliers.add(() -> {
                List<double[]> dists = new ArrayList<>(nv);
                int lim = sv + nv;
                for (int j = sv; j < lim; j++) {
                    double[] vec = vectors.get(j);
                    double[] d = new double[numCenters];
                    for (int k = 0; k < numCenters; k++) {
                        d[k] = euclideanDistance(vec, centers.get(k));
                    }
                    dists.add(d);
                }
                return dists;
            });

            startVector += nv;
        }

        List<Tuple2<Integer, List<double[]>>> tuples =
                Flux.zip(Flux.range(0, distanceSuppliers.size()), Flux.fromIterable(distanceSuppliers))
                        .flatMap(tuple -> Mono.just(tuple)
                                .subscribeOn(Schedulers.parallel())
                                .map(tpl -> Tuples.of(tpl.getT1(), tpl.getT2().get())))
                        .subscribeOn(Schedulers.immediate())
                        .collectList()
                        .block();

        // Since the computations were done on multiple threads, they'll usually be out of order.
        tuples.sort((tpl1, tpl2) -> tpl1.getT1().compareTo(tpl2.getT1()));

        List<double[]> distances = new ArrayList<>(numVectors);
        tuples.forEach(tpl -> distances.addAll(tpl.getT2()));

        return distances;
    }

    public static List<double[]> computeDistancesInParallelUsingParallelStreams(
            List<double[]> vectors,
            List<double[]> centers) {

        int numVectors = vectors.size();
        int numCenters = centers.size();

        // Create the distances list and prepopulate with zero vectors.
        List<double[]> distances = new ArrayList<>(numVectors);
        for (int i = 0; i < numVectors; i++) {
            distances.add(new double[numCenters]);
        }

        IntStream.range(0, vectors.size())
                // This causes forEach to be called on multiple threads simultaneous. The Intstream is necessary
                // to preserve the correct order of the distances in the result.
                .parallel()
                .forEach(n -> {
                    double[] vec = vectors.get(n);
                    double[] dists = distances.get(n);
                    for (int i = 0; i < numCenters; i++) {
                        dists[i] = euclideanDistance(vec, centers.get(i));
                    }
                });

        return distances;
    }

    public static List<double[]> computesDistancesInParallelUsingCompletableFutures(
            List<double[]> vectors,
            List<double[]> centers,
            Executor executor
    ) throws ExecutionException, InterruptedException {

        final int numVectors = vectors.size();
        final int numCenters = centers.size();

        final int[] vectorsPerThread = vectorsPerThread(numVectors);
        final int numThreads = vectorsPerThread.length;

        List<CompletableFuture<List<double[]>>> completableFutures = new ArrayList<>(numThreads);
        int startVector = 0;
        for (int i = 0; i < numThreads; i++) {
            final int sv = startVector;
            final int nv = vectorsPerThread[i];
            completableFutures.add(CompletableFuture.supplyAsync(() -> {
                List<double[]> dists = new ArrayList<>(nv);
                int lim = sv + nv;
                for (int j = sv; j < lim; j++) {
                    double[] d = new double[numCenters];
                    for (int k = 0; k < numCenters; k++) {
                        d[k] = euclideanDistance(vectors.get(j), centers.get(k));
                    }
                    dists.add(d);
                }
                return dists;
            }, executor));
            startVector += nv;
        }

        CompletableFuture<Void> allFutures = CompletableFuture.allOf(
                completableFutures.toArray(new CompletableFuture[numThreads])
        );

        CompletableFuture<List<List<double[]>>> futuresResult = allFutures.thenApply(
                v -> completableFutures.stream().map(CompletableFuture::join).collect(Collectors.toList())
        );

        List<double[]> finalResults = new ArrayList<>(vectors.size());
        futuresResult.get().forEach(lst -> finalResults.addAll(lst));

        return finalResults;
    }

    public static List<double[]> computeDistancesUsingCallables(
            List<double[]> vectors,
            List<double[]> centers,
            ExecutorService executorService) throws InterruptedException, ExecutionException {

        final int numVectors = vectors.size();
        final int numCenters = centers.size();

        final int[] vectorsPerThread = vectorsPerThread(numVectors);
        final int numThreads = vectorsPerThread.length;

        List<Callable<List<double[]>>> callables = new ArrayList<>(numThreads);
        int startVector = 0;
        for (int i = 0; i < numThreads; i++) {
            final int sv = startVector;
            final int nv = vectorsPerThread[i];

            callables.add(() -> {
                List<double[]> dists = new ArrayList<>(nv);
                int lim = sv + nv;
                for (int j = sv; j < lim; j++) {
                    double[] d = new double[numCenters];
                    for (int k = 0; k < numCenters; k++) {
                        d[k] = euclideanDistance(vectors.get(j), centers.get(k));
                    }
                    dists.add(d);
                }
                return dists;
            });

            startVector += nv;
        }

        List<Future<List<double[]>>> futures = executorService.invokeAll(callables);
        List<double[]> distances = new ArrayList<>(numVectors);

        for (Future<List<double[]>> future : futures) {
            distances.addAll(future.get());
        }

        return distances;
    }

    /**
     * Given a fixed number of vectors to compute distances for, this method returns an array
     * equal to the number of threads, with each element being the number of vectors handled by each thread.
     *
     * @param numVectors
     *
     * @return an array of length equal to the number of threads to use. The elements are the number
     *   of vectors to handle for the thread corresponding to that index. All elements will be 1 or greater.
     */
    public static int[] vectorsPerThread(int numVectors) {
        // No point in having the number of threads greater than the number of vectors.
        int numThreads = Math.min(numVectors, Runtime.getRuntime().availableProcessors());
        int vpt = numVectors/numThreads;
        int[] vectorsPerThread = new int[numThreads];
        Arrays.fill(vectorsPerThread, vpt);
        // numThreads might not be evenly divisible into numVectors, so some threads will need to
        // handle an additional vector.
        int leftOver = numVectors%numThreads;
        for (int i=0; i<leftOver; i++) {
            vectorsPerThread[i]++;
        }
        return vectorsPerThread;
    }

    public static List<double[]> generateRandomVectors(int numVectors, int vectorLength, long seed) {
        if (seed == -1L) {
            seed = System.currentTimeMillis();
        }
        Random random = new Random();
        List<double[]> vectors = new ArrayList<>(numVectors);
        for (int i = 0; i < numVectors; i++) {
            double[] vec = new double[vectorLength];
            for (int j = 0; j < vectorLength; j++) {
                vec[j] = random.nextDouble();
            }
            vectors.add(vec);
        }
        return vectors;
    }

    public static double euclideanDistance(double[] vector1, double[] vector2) {
        double sum = 0.0;
        for (int i = 0; i < vector1.length; i++) {
            double d = vector1[i] - vector2[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    public static boolean checkDistancesEqual(String groundTruthName, Map<String, List<double[]>> distMap) {
        List<double[]> trueDistances = distMap.get(groundTruthName);
        AtomicBoolean equal = new AtomicBoolean(true);
        distMap.entrySet().forEach(e -> {
            if (!groundTruthName.equals(e.getKey())) {
                if (!isEqual(trueDistances, e.getValue())) {
                    System.out.printf("%s: distances not equal to distances computed by %s%n",
                            e.getKey(), groundTruthName);
                    equal.set(false);
                }
            }
        });
        return equal.get();
    }

    public static boolean isEqual(List<double[]> vectors1, List<double[]> vectors2) {
        final int numVectors = vectors1.size();
        if (vectors2.size() != numVectors) {
            return false;
        }
        for (int i = 0; i < numVectors; i++) {
            if (!Arrays.equals(vectors1.get(i), vectors2.get(i))) {
                return false;
            }
        }
        return true;
    }
}
