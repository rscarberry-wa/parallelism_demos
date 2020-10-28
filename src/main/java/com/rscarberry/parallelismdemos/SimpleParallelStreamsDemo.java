package com.rscarberry.parallelismdemos;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * The purpose of this program is to verify whether using parallelStream on a list then
 * transforming the elements and collecting them into a new list preserves their order. It does, at
 * least on the Oracle Java 11 JDK than I'm using. -- R.Scarberry 10/27/2020.
 */
public class SimpleParallelStreamsDemo {

    public static void main(String[] args) {

        // Words to process. Used "one", "two", etc., so it'd be easier to see whether order is preserved.
        List<String> words = List.of("one two three four five six seven eight nine ten".split(" "));

        // In order to see which threads process which words.
        Map<String, List<String>> threadMap = new ConcurrentHashMap<>();

        List<String> ucWords = words.parallelStream()
                .map(s -> { // Just uppercase the words, but add a random sleep to simulate a time-consuming op.
                    String threadName = Thread.currentThread().getName();
                    // Update the thread map.
                    threadMap.compute(threadName, (tn, wordList) -> {
                        if (wordList == null) {
                            wordList = new ArrayList<>();
                        }
                        wordList.add(s);
                        return wordList;
                    });
                    // Add a small random sleep of [0 - 50) msec.
                    try {
                        Thread.sleep((long)(Math.random() * 50.0));
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    return s.toUpperCase();
                })
                .collect(Collectors.toList());

        System.out.println("Resulting words:");
        for (String ucWord: ucWords) {
            System.out.printf("\t%s%n", ucWord);
        }

        System.out.println("\nThreads and the words processed by them:");
        for (Map.Entry<String, List<String>> entry: threadMap.entrySet()) {
            System.out.printf("\t%-35s: %s%n", entry.getKey(), entry.getValue());
        }
    }

}
