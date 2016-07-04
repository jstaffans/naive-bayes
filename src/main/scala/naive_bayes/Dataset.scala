package naive_bayes

import scalaj.http._
import scala.util.Random

object Dataset {

  /**
    * Loads dataset into a sequence of feature vectors.
    */
  def load(): Seq[Vector[Double]] = {
    val response: HttpResponse[String] =
      Http("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")
        .asString

    val trim = (s: String) => s.trim
    val parse = (s: String) => s.toDouble
    val convert = parse compose trim
    val parseLine = (line: String) => line.split(",").map(convert).toVector

    response.body.lines.map(parseLine).toSeq
  }

  /**
   * Splits data into training and test data sets.
   */
  def split(data: Seq[Vector[Double]], splitRatio: Double): (Seq[Vector[Double]], Seq[Vector[Double]]) = {
    val trainSize = Math.round(data.length * splitRatio).toInt
    val shuffled = Random.shuffle(data).toArray
    (shuffled.take(trainSize), shuffled.drop(trainSize))
  }

  /**
   * Groups samples by class (which is the last element in the feature vector)
   */
  def separate(data: Seq[Vector[Double]]): Map[Double, Seq[Vector[Double]]] = {
    data.groupBy(_.last)
  }

  /**
    * Provides a summary for a given feature (mean and N-1 standard deviation)
    */
  private def summarizeFeature(samples: Seq[Double]): (Double, Double) = {
    val mean = samples.sum / samples.size.toDouble
    val variance = samples.map(sample => Math.pow(sample - mean, 2.0)).sum / (samples.size - 1).toDouble
    val stdev = Math.sqrt(variance)
    (mean, stdev)
  }

  /**
    * Summarizes data by class (mean and standard deviation for each feature)
    */
  def summarize(data: Seq[Vector[Double]]): Map[Double, Seq[(Double, Double)]] = {
    separate(data).map {case (k, v) =>
      val features = v.transpose.dropRight(1)   // classification is not a feature -- drop it
      (k, features.map(summarizeFeature))
    }
  }

  /**
    * Gaussian probability of one data point.
    */
  def probability(x: Double, mean: Double, stdev: Double): Double = {
    val exponent = Math.exp(-(Math.pow(x - mean, 2.0)/(2 * Math.pow(stdev, 2))))
    (1 / (Math.sqrt(2 * Math.PI) * stdev)) * exponent
  }

  /**
    * Produce a map of classIds to probability of belonging to that class
    * for a given input vector
    */
  def classProbabilities(summaries: Map[Double, Seq[(Double, Double)]], input: Vector[Double]): Map[Double, Double] = {
    summaries.map {case (classId, classSummaries) =>
      val p: Vector[(Double, (Double, Double))]  = input zip classSummaries

      // multiply all the probabilities together
      (classId, p.foldLeft(1.0) { (acc, curr) =>
        val (input, (mean, stdev)) = curr
        acc * probability(input, mean, stdev)
      })
    }
  }

  def predict(summaries: Map[Double, Seq[(Double, Double)]], input: Vector[Double]): Double = {
    val probabilities = classProbabilities(summaries, input)
    val prediction = probabilities.toList.sortBy {_._2}
    prediction.last._1
  }
}

