// :load /home/siddharth/vscode/class_work/BDA/curly-octo-system/src/main/scala/thefinalthingig.scala


print("\033c" )

import org.apache.spark.{SparkConf, SparkContext}
import java.io.FileWriter
import java.io.BufferedWriter
import org.apache.spark.mllib.stat.Statistics

val outputFile = "src/main/resources/output/output.txt"
val writer = new BufferedWriter(new FileWriter(outputFile, true))


case class Flight(dofM: String, dofW: String, carrier: String, tailnum: String, flnum: Int, org_id: String, origin: String,
                    dest_id: String, dest: String, crsdeptime: Double, deptime: Double, depdelaymins: Double, crsarrtime: Double,
                    arrtime: Double, arrdelay: Double, crselapsedtime: Double, dist: Int)

def parseFlight(str: String): Flight = {
    val line = str.split(",")
    Flight(line(0), line(1), line(2), line(3), line(4).toInt, line(5), line(6), line(7), line(8), line(9).toDouble,
    line(10).toDouble, line(11).toDouble, line(12).toDouble, line(13).toDouble, line(14).toDouble, line(15).toDouble, line(16).toInt)
}


// ------------------------------------- initializes a local Spark context -------------------------------------
val sparkConf = new SparkConf()
.setAppName("FlightDelayPrediction") // spark submitted app name
.setMaster("local[*]") // run spark using all available cores
val sc = new SparkContext(sparkConf)
// ------------------------------------- ------------------------------------- -------------------------------------


// ------------------------------------- read csv file -------------------------------------
val data = sc.textFile("src/main/resources/dataset/flight_data.csv")

// remove header
val header = data.first()
val textRDD = data.filter(row => row != header)
val flightsRDD = textRDD.map(parseFlight).cache()
// ------------------------------------- ------------------------------------- -------------------------------------




// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



/* ------------------------------------- -------------------------------------
                      small data analysis on the dataset 
  ------------------------------------- ------------------------------------- */


// Calculate total number of flights

val totalFlights = flightsRDD.count()
writer.append("Total number of flights: " + totalFlights + "\n\n")


// ------------------------------------- Num of flights per origin terminal -------------------------------------

val originCounts = flightsRDD.map(flight => (flight.origin, 1)).reduceByKey(_ + _).sortBy(_._2 , ascending = false)

writer.append("\n\nNum of flights per origin terminal \n")
writer.append("Origin\t\tCount\n")
writer.append("---------------------\n")
originCounts.take(5).foreach { case (origin, count) =>
  writer.append(s"$origin\t\t$count\n")
}


// ------------------------------------- Num of flights per destination -------------------------------------

val destCounts = flightsRDD.map(flight => (flight.dest, 1)).reduceByKey(_ + _).sortBy(_._2 , ascending = false)

writer.append("\n\nNum of flights per destination \n")
writer.append("dest\t\tCount\n")
writer.append("---------------------\n")
destCounts.take(5).foreach { case (dest, count) =>
  writer.append(s"$dest\t\t$count\n")
}


/* ------------------------------------- -------------------------------------
                Speed and Time based analysis on the dataset 
  ------------------------------------- ------------------------------------- */


// ------------------------------------- Average Speed, Standard deviation -------------------------------------


writer.append("\n\n------------------------------------- -------------------------------------\n")
writer.append("              Speed and Time based analysis on the dataset\n")
writer.append("------------------------------------- -------------------------------------\n")


val averageSpeed = flightsRDD.map(flight => (flight.tailnum, flight.dist.toDouble / (flight.crselapsedtime.toDouble / 60))) // Find the average speed of each flight (tail number)

val avgSpeedByTailNum_per_flight = averageSpeed.groupByKey().mapValues(speeds => speeds.sum / speeds.size) // Group by tail number and find the average speed


// print 5 rows
writer.append("\n\nAverage speed of each flight (tail number) \n")
writer.append("Tail Number\t\tAverage Speed\n")
writer.append("------------------------------------------\n")
avgSpeedByTailNum_per_flight.take(5).foreach { case (tailnum, speed) =>
  writer.append(s"$tailnum\t\t$speed\n")
}

// Group by tail number and find the standard deviation of the average speed
val stdDevByTailNum = averageSpeed.groupByKey().mapValues(speeds => {
  val avg = speeds.sum / speeds.size
  val squaredDiffs = speeds.map(speed => math.pow(speed - avg, 2))
  math.sqrt(squaredDiffs.sum / squaredDiffs.size)
})

// print 5 rows
writer.append("\n\nStandard deviation of the average speed of each flight (tail number) \n")
writer.append("Tail Number\t\tStandard deviation\n")
writer.append("------------------------------------------\n")
stdDevByTailNum.take(5).foreach { case (tailnum, stdDev) =>
  writer.append(s"$tailnum\t\t$stdDev\n")
}


// ------------------------------------- Correlation between different metrics -------------------------------------



// corelation between distance and speed total
val distance = flightsRDD.map(flight => flight.dist.toDouble)
val speed = flightsRDD.map(flight => flight.dist.toDouble / (flight.crselapsedtime.toDouble / 60))

val dist_to_speed_corr = Statistics.corr(distance, speed)

writer.append("\n\nCorrelation between distance and speed \n")
writer.append("------------------------------------------\n")
writer.append("The corelation between distance to speed is = " + dist_to_speed_corr.toString + "\n")


// corelation with respect to the arrival delay and departure delay
val arrivalDelay = flightsRDD.map(flight => flight.arrdelay.toDouble)
val departureDelay = flightsRDD.map(flight => flight.depdelaymins.toDouble)

val arrival_to_departure_corr = Statistics.corr(arrivalDelay, departureDelay)

writer.append("\n\nCorrelation between arrival delay and departure delay \n")
writer.append("------------------------------------------\n")
writer.append("The corelation between arrival delay and departure delay is = " + arrival_to_departure_corr.toString + "\n")








/* ------------------------------------- -------------------------------------
                                Overall and Misc 
  ------------------------------------- ------------------------------------- */

writer.append("\n\n------------------------------------- -------------------------------------\n")
writer.append("                                 Overall and Misc\n")
writer.append("------------------------------------- -------------------------------------\n")


//  Num of Planes in each day of week
val planesPerDayOfWeek = flightsRDD.groupBy(_.dofW).mapValues(_.size).sortBy(_._1 , ascending = true)

writer.append("\n\nNum of Planes in each day of week \n")
writer.append("Day of Week\t\tCount\n")
writer.append("---------------------\n")
planesPerDayOfWeek.take(7).foreach { case (dofW, count) =>
  writer.append(s"$dofW\t\t$count\n")
}

// Average delay in each day of week in hours
val average_delay_per_week = flightsRDD.groupBy(_.dofW).mapValues(_.map(_.depdelaymins).sum / 60).sortBy(_._1 , ascending = true)

writer.append("\n\nAverage delay in each day of week in hours \n")
writer.append("Day of Week\t\tAverage Delay (in hours)\n")
writer.append("---------------------\n")
average_delay_per_week.take(7).foreach { case (dofW, avg_delay) =>
  writer.append(s"$dofW\t\t$avg_delay\n")
}

// overall delay percentage
val total_number_of_flights = flightsRDD.count()
val total_number_of_delayed_flights = flightsRDD.filter(flight => flight.depdelaymins > 20).count()
val overall_delay_percentage = (total_number_of_delayed_flights.toDouble / total_number_of_flights.toDouble) * 100

writer.append("\n\nOverall delay percentage \n")
writer.append("Overall delay percentage = " + overall_delay_percentage.toString + "\n")


// Carrier wise percent of delays
val carrier_wise_delay_percentage = flightsRDD.map(flight => (flight.carrier, flight.depdelaymins)).filter(flight => flight._2 > 20).map(flight => (flight._1, 1)).reduceByKey(_ + _).mapValues(_ / total_number_of_delayed_flights.toDouble * 100).sortBy(_._2 , ascending = false)

writer.append("\n\nCarrier wise percent of delays \n")
writer.append("Carrier\t\tPercentage of delay\n")
writer.append("---------------------\n")
carrier_wise_delay_percentage.take(10).foreach { case (carrier, percentage) =>
  writer.append(s"$carrier\t\t$percentage\n")
}


writer.close()