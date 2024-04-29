using Microsoft.ML.Data;

namespace StockMarketPrediction.DataStructures;

public class StockPrice
{
    [LoadColumn(0)]
    public string Date;
    [LoadColumn(1)]
    public float Open;
    [LoadColumn(2)]
    public float High;
    [LoadColumn(3)]
    public float Low;
    [LoadColumn(4)]
    public float Close;
    [LoadColumn(5)]
    public float AdjClose;
    [LoadColumn(6)]
    public Int64 Volume;
    
}

class Data
{

    public float Open;
    public float Close;
    public float High;
    public float Low;
    public float AdjClose;

}

class ForecastResult
{
    public float[] Forecast { get; set; }
}

class BoundedForecastResult
{
    public float[] Forecast { get; set; }
    public float[] LowerBound { get; set; }
    public float[] UpperBound { get; set; }
}