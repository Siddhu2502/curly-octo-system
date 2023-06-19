print("\033c" )

// :load /home/siddharth/vscode/class_work/BDA/curly-octo-system/src/main/scala/flightclassification.scala

import org.apache.spark._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkConf, SparkContext}

case class Flight(dofM: String, dofW: String, carrier: String, tailnum: String, flnum: Int, org_id: String, origin: String,
                    dest_id: String, dest: String, crsdeptime: Double, deptime: Double, depdelaymins: Double, crsarrtime: Double,
                    arrtime: Double, arrdelay: Double, crselapsedtime: Double, dist: Int, arrdelaymins: Double)

def parseFlight(str: String): Flight = {
    val line = str.split(",")
    Flight(line(0), line(1), line(2), line(3), line(4).toInt, line(5), line(6), line(7), line(8), line(9).toDouble,
    line(10).toDouble, line(11).toDouble, line(12).toDouble, line(13).toDouble, line(14).toDouble, line(15).toDouble, line(16).toInt, line(14).toDouble)
}


// ------------------------------------- initializes a local Spark context -------------------------------------
val sparkConf = new SparkConf()
.setAppName("FlightDelayPrediction") // spark submitted app name
.setMaster("local[*]") // run spark using all available cores
val sc = new SparkContext(sparkConf)

// read csv file
val data = sc.textFile("src/main/resources/dataset/flight_data.csv")

// remove header
val header = data.first()
val textRDD = data.filter(row => row != header)

val flightsRDD = textRDD.map(parseFlight).cache()



// next step is to transform the non-numeric features into numeric values

// transform non-numeric carrier value to numeric value
var carrierMap: Map[String, Int] = Map()
var index: Int = 0
flightsRDD.map(flight => flight.carrier).distinct.collect.foreach(x => { carrierMap += (x -> index); index += 1 })
// Print carrierMap
println("Carrier Map:")
carrierMap.foreach(println)


//  ------------------------------------- origin map -------------------------------------

// transform non-numeric origin value to numeric value
var originMap: Map[String, Int] = Map()
var index1: Int = 0
flightsRDD.map(flight => flight.origin).distinct.collect.foreach(x => { originMap += (x -> index1); index1 += 1 })


//  ------------------------------------- destMap map -------------------------------------

// transform non-numeric destination value to numeric value
var destMap: Map[String, Int] = Map()
var index2: Int = 0
flightsRDD.map(flight => flight.dest).distinct.collect.foreach(x => { destMap += (x -> index2); index2 += 1 })

// ------------------------------------- creating the features array -------------------------------------
val feature_array = flightsRDD.map(flight => {
    val monthday = flight.dofM.toInt - 1 // category // -1 because feature starts with 0
    val weekday = flight.dofW.toInt - 1 // category // -1 because feature starts with 0
    val crsdeptime1 = flight.crsdeptime.toInt
    val crsarrtime1 = flight.crsarrtime.toInt
    val carrier1 = carrierMap(flight.carrier) // category
    val crselapsedtime1 = flight.crselapsedtime
    val delayed = if (flight.depdelaymins > 40) 1.0 else 0.0
    Array(delayed.toDouble, monthday.toDouble, 
            weekday.toDouble, crsdeptime1.toDouble, crsarrtime1.toDouble, 
            carrier1.toDouble, crselapsedtime1.toDouble)
    })

// Calculate total number of flights
val totalFlights = flightsRDD.count()
print(totalFlights)


// ------------------------------------- Num of flights per terminal -------------------------------------

val flightsPerTerminal = flightsRDD.map(flight => (flight.origin, 1)).reduceByKey(_ + _)
println("\nNumber of flights per terminal: \n")
println("Key\tValue")
flightsPerTerminal.foreach{
    case (key,value) => println(key + "\t" + value)
}


// ------------------------------------- Calculate the average departure delay for each airport -------------------------------------
val avgDepartureDelaysPerAirport = flightsRDD.map(flight => (flight.origin, flight.depdelaymins)).groupByKey().mapValues(avg => avg.sum/avg.size)
println("\nAverage departure delay per airport: \n")
println("Key\tValue")
avgDepartureDelaysPerAirport.foreach{
    case (key,value) => println(key + "\t" + value)
}


// ------------------------------------- Calculate the average arrival delay for each airport -------------------------------------
val avgArrivalDelaysPerAirport = flightsRDD.map(flight => (flight.origin, flight.arrdelaymins)).groupByKey().mapValues(avg => avg.sum/avg.size)
println("\nAverage departure delay per airport: \n")
println("Key\tValue")
avgArrivalDelaysPerAirport.foreach{
    case (key,value) => println(key + "\t" + value)
}


// ------------------------------------- Count of delayed flights (departure and arrival) -------------------------------------

val departure_delay_count = flightsRDD.filter(flight => flight.depdelaymins > 20).count()
val arrival_delay_count = flightsRDD.filter(flight => flight.arrdelaymins > 20).count()
val total_count = departure_delay_count + arrival_delay_count
print("\nTotal number of delayed flights: " + total_count + "\n")


// ------------------------------------- flights that were delayed both at departure and arrival -------------------------------------

val delayed_flight_atdepandarr = flightsRDD.filter(flight => flight.depdelaymins > 20 && flight.arrdelaymins > 20).count()
print("\nTotal number of flights delayed at diparture and arrival: " + total_count + "\n")


// ------------------------------------- flights that were not delayed both at departure and arrival -------------------------------------

val delayed_flight_atdepandarr = flightsRDD.filter(flight => flight.depdelaymins == 0 && flight.arrdelaymins == 20).count()
print("\nTotal number of flights not delayed at diparture and arrival: " + total_count + "\n")


// ------------------------------------- Top 3 origin airport code delay with total flights and on-time flights count -------------------------------------

val top_origin_delay = flightsRDD.map(flight => (flight.origin, flight.depdelaymins)).reduceByKey(_ + _).sortBy(_._2, false).take(3)
println("Top 3 origin airport codes with the highest delays:")
val origin_delay_df = top_origin_delay.map{ case (code, delay) => 
    val total_flights = flightsRDD.filter(flight => flight.origin == code).count()
    val delayed_flights = flightsRDD.filter(flight => flight.origin == code && flight.depdelaymins > 0).count()
    (code, total_flights, delayed_flights)
}
println("%-13s %-14s %-16s".format("Airport Code\tTotal Flights\tDelayed Flights"))
origin_delay_df.foreach{ case (code, total_flights, delayed_flights) => 
    println("%-13s %-14s %-16s".format(code + "\t" + total_flights + "\t" + delayed_flights))
}


// ------------------------------------- Top 3 departure airport code delay with total flights and on-time flights count -------------------------------------

val top_departure_delay = flightsRDD.map(flight => (flight.dest, flight.arrdelaymins)).reduceByKey(_ + _).sortBy(_._2, false).take(3)
println("Top 3 departure airport codes with the highest delays:")
val departure_delay_df = top_departure_delay.map { case (code, delay) =>
  val total_flights = flightsRDD.filter(flight => flight.dest == code).count()
  val delayed_flights = flightsRDD.filter(flight => flight.dest == code && flight.arrdelaymins > 0).count()
  (code, total_flights, delayed_flights)
}

println("%-13s %-14s %-16s".format("Airport Code", "Total Flights", "Delayed Flights"))
departure_delay_df.foreach { case (code, total_flights, delayed_flights) =>
  println("%-13s %-14s %-16s".format(code, total_flights, delayed_flights))
}


// ------------------------------------- Top 3 origin airport code on time flights with total flights and on-time flights count -------------------------------------

println("Top 3 origin airport codes with the highest on-time flights:")
val top_origin_ontime = flightsRDD.map(flight => (flight.origin, flight.depdelaymins)).reduceByKey(_ + _).sortBy(_._2, false).take(3)
val origin_ontime_df = top_origin_ontime.map { case (code, delay) =>
  val total_flights = flightsRDD.filter(flight => flight.origin == code).count()
  val ontime_flights = flightsRDD.filter(flight => flight.origin == code && flight.depdelaymins == 0).count()
  (code, total_flights, ontime_flights)
}

println("%-13s %-14s %-16s".format("Airport Code", "Total Flights", "On-time Flights"))
origin_ontime_df.foreach { case (code, total_flights, ontime_flights) =>
  println("%-13s %-14s %-16s".format(code, total_flights, ontime_flights))
}

// ------------------------------------- Top 3 departure airport code on time flights with total flights and on-time flights count -------------------------------------

println("Top 3 departure airport codes with the highest on-time flights:")
val top_departure_ontime = flightsRDD.map(flight => (flight.dest, flight.arrdelaymins)).reduceByKey(_ + _).sortBy(_._2, false).take(3)
val departure_ontime_df = top_departure_ontime.map { case (code, delay) =>
  val total_flights = flightsRDD.filter(flight => flight.dest == code).count()
  val ontime_flights = flightsRDD.filter(flight => flight.dest == code && flight.arrdelaymins == 0).count()
  (code, total_flights, ontime_flights)
}

println("%-13s %-14s %-16s".format("Airport Code", "Total Flights", "On-time Flights"))
departure_ontime_df.foreach { case (code, total_flights, ontime_flights) =>
  println("%-13s %-14s %-16s".format(code, total_flights, ontime_flights))
}



// ------------------------------------- Top planes with highest average departure delays -------------------------------------

val top_planes_highest_dept_delays = flightsRDD.map(flight => (flight.tailnum, flight.depdelaymins)).groupByKey().mapValues(avg => avg.sum/avg.size.toDouble).sortBy(_._2, false).take(3)
println("Top 3 planes with the highest average departure delays:")
top_planes_highest_dept_delays.foreach{ case (tailnum, delay) => println(tailnum + "\t" + delay) }
println()


// ------------------------------------- Top planes with lowest average departure delays -------------------------------------

val top_planes_lowest_dept_delays = flightsRDD.map(flight => (flight.tailnum, flight.depdelaymins)).groupByKey().mapValues(avg => avg.sum/avg.size.toDouble).sortBy(_._2).take(3)
println("Top 3 planes with the lowest average departure delays:")
top_planes_lowest_dept_delays.foreach{ case (tailnum, delay) => println(tailnum + "\t" + delay) }
println()


// ------------------------------------- Top planes with highest average arrival delays -------------------------------------

val top_planes_highest_arrival_delays = flightsRDD.map(flight => (flight.tailnum, flight.arrdelaymins)).groupByKey().mapValues(avg => avg.sum/avg.size.toDouble).sortBy(_._2, false).take(3)
println("Top 3 planes with the highest average arrival delays:")
top_planes_highest_arrival_delays.foreach{ case (tailnum, delay) => println(tailnum + "\t" + delay) }
println()


// ------------------------------------- Top planes with lowest average arrival delays -------------------------------------

val top_planes_lowest_arrival_delays = flightsRDD.map(flight => (flight.tailnum, flight.arrdelaymins)).groupByKey().mapValues(avg => avg.sum/avg.size.toDouble).sortBy(_._2).take(3)
println("Top 3 planes with the lowest average arrival delays:")
top_planes_lowest_arrival_delays.foreach{ case (tailnum, delay) => println(tailnum + "\t" + delay) }
println()











// -------------------------------------------------------------- its ml part dont mind --------------------------------------------------------------

// // generating Labeled Points
// // first parameter is label or target variable which is 'delayed' in our case
// // second parameter is a vector of features
// val mldata = mlprep.map(x => LabeledPoint(x(0), Vectors.dense(x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8))))

// // split the data into training and test data set

// // mldata0 is 85% not delayed flights
// val mldata0 = mldata.filter(x => x.label == 0).randomSplit(Array(0.85, 0.15))(0)
// // mldata1 is %100 delayed flights
// val mldata1 = mldata.filter(x => x.label != 0)
// // mldata2 is mix of delayed and not delayed
// val mldata2 = mldata0 ++ mldata1

// // split mldata2 into training and test data
// val splits = mldata2.randomSplit(Array(0.7, 0.3))
// val (trainingData, testData) = (splits(0), splits(1))

// // next step is to train the model

// /* categoricalFeaturesInfo specifies which features are categorical and how many categorical values each of those features can take.
//  This is given as a map from feature index to the number of categories for that feature.*/
// var categoricalFeaturesInfo = Map[Int, Int]()
// categoricalFeaturesInfo += (0 -> 31) // day of month
// categoricalFeaturesInfo += (1 -> 7)  // day of week
// categoricalFeaturesInfo += (4 -> carrierMap.size)
// categoricalFeaturesInfo += (6 -> originMap.size)
// categoricalFeaturesInfo += (7 -> destMap.size)

// val numClasses = 2 //delayed(1) and not-delayed(0)
// val impurity = "gini"
// val maxDepth = 10
// val maxBins = 5000

// val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

// // print the decision tree
// println(model.toDebugString)

// // test the model
// val labelAndPreds = testData.map { point =>
//     val prediction = model.predict(point.features)
//     (point.label, prediction)
// }

// // calculate wrong and correct prediction percentage

// val wrongPrediction = labelAndPreds.filter{
//     case (label, prediction) => label != prediction
// }

// val wrongCount = wrongPrediction.count()

// val correctPrediction = labelAndPreds.filter{
//     case (label, prediction) => label == prediction
// }

// val correctCount = correctPrediction.count()

// println("Wrong Count: " + wrongCount)
// println("Wrong Percentage: " + (wrongCount.toDouble/testData.count()) * 100)
// println("Correct Count: " + correctCount)
// println("Correct Percentage: " + (correctCount.toDouble/testData.count()) * 100)