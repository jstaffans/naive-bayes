package naive_bayes

/**
  * Author: johannes.
  */
object Diabetes {

  def main(args: Array[String]) {
    val data = Dataset.load()
    val (trainSet, testSet) = Dataset.split(data, 0.33)
    val summaries = Dataset.summarize(trainSet)

    val testSetPredictions = testSet.map(Dataset.predict(summaries, _))

    val numAccurate = (testSet zip testSetPredictions) map { case (testSample, predictedClass) =>
      if (testSample.last == predictedClass) 1 else 0
    } sum

    val accuracy = 100.0 * numAccurate / testSet.size

    println(s"Accuracy: $accuracy %")
  }
}
