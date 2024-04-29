using Microsoft.ML.Data;

namespace DemandForecasting.DataStructures;

public class Rental
{

    [LoadColumn(1)]
    public DateTime RentalDate;
    [LoadColumn(2)]
    public int Season;
    [LoadColumn(10)]
    public double Temp;
    [LoadColumn(15)]
    public int TotalRentals;
}

public class TrialRental
{

    [LoadColumn(0)]
    public int Indexer;
    [LoadColumn(1)]
    public DateTime RentalDate;
    [LoadColumn(2)]
    public int Season;
    [LoadColumn(3)]
    public int Year;
    [LoadColumn(4)]
    public int Month;
    [LoadColumn(5)]
    public int Holiday;
    [LoadColumn(6)]
    public int Weekday;
    [LoadColumn(7)]
    public int WorkDay;
    [LoadColumn(8)]
    public int Weather;
    [LoadColumn(9)]
    public float WeatherTemp;
    [LoadColumn(10)]
    public float TempAdj;
    [LoadColumn(11)]
    public float Humidity;
    [LoadColumn(12)]
    public float Wind;
    [LoadColumn(13)]
    public int Casual;
    [LoadColumn(14)]
    public int Registered;
    [LoadColumn(15)]
    public int TotalRentals;
}