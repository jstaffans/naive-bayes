package naive_bayes

import org.scalatest._

/**
  * Author: johannes.
  */
class DatasetSuite extends FunSuite with Matchers {

  test("A dataset can be splitted into training and test sets") {
    val data = Seq(
      Vector(1.0),
      Vector(2.0),
      Vector(3.0),
      Vector(4.0),
      Vector(5.0))

    val (train, test) = Dataset.split(data, 0.33)

    assert(train.length == 2)
    assert(test.length == 3)

    // check that sets don't overlap
    train.foreach(trainVal => {
      test.foreach(testVal => {
        assert(testVal(0) != trainVal(0))
      })
    })
  }

  test("Dataset can be grouped by class") {
    val data = Seq(
      Vector(1.0, 0.0),
      Vector(2.0, 0.0),
      Vector(3.0, 1.0),
      Vector(4.0, 1.0),
      Vector(5.0, 1.0))

    val separated = Dataset.separate(data)

    assert(separated.get(0.0).size == 2)
    assert(separated.get(1.0).size == 3)
  }

  test("Summarized data set (means and standard deviations)") {
    val data = Seq(
      Vector(1.0, 10.0, 0.0),
      Vector(2.0, 12.0, 0.0),
      Vector(3.0, 14.0, 0.0),
      Vector(4.0, 16.0, 0.0),
      Vector(5.0, 18.0, 0.0))

    val summary = Dataset.summarize(data)
    assert(summary.contains(0.0))
    val feature1 = summary.getOrElse(0.0, List()).head

    val (mean, stdev) = (feature1._1, feature1._2)
    assert(mean == 3.0)
    assert(stdev > 1.5)
    assert(stdev < 1.6)
  }

  test("Probability") {
    val data = Seq(
      Vector(1.0, 10.0, 0.0),
      Vector(2.0, 12.0, 0.0),
      Vector(3.0, 14.0, 0.0),
      Vector(40.0, 160.0, 1.0),
      Vector(50.0, 180.0, 1.0))

    val summaries = Dataset.summarize(data)

    val classification = Dataset.classProbabilities(summaries, Vector(1.0, 5.0))

    println(classification)
  }

  test("Prediction") {
    val data = Seq(
      Vector(1.0, 10.0, 0.0),
      Vector(2.0, 12.0, 0.0),
      Vector(3.0, 14.0, 0.0),
      Vector(40.0, 160.0, 1.0),
      Vector(50.0, 180.0, 1.0))

    val summaries = Dataset.summarize(data)

    assert(Dataset.predict(summaries, Vector(4.0, 11.0)) == 0.0)
    assert(Dataset.predict(summaries, Vector(60.0, 170.0)) == 1.0)
  }
}
